import asyncio
import os
import random
from datetime import datetime
from pathlib import Path

from langchain.callbacks import get_openai_callback

from chatbot.ai.workers.video_chatter_summary_builder.video_chatter_summary_builder import VideoChatterSummaryBuilder
from chatbot.mongo_database.mongo_database_manager import MongoDatabaseManager
from chatbot.system.filenames_and_paths import get_thread_backups_collection_name, \
    VIDEO_CHATTER_SUMMARIES_COLLECTION_NAME


async def generate_video_chatter_summaries(mongo_database: MongoDatabaseManager,
                                           thread_collection_name: str,
                                           video_chatter_summaries_collection_name: str,
                                           designated_channel_name: str = "introductions",
                                           use_anthropic: bool = False,
                                           overwrite: bool = False,
                                           ):
    thread_collection = mongo_database.get_collection(thread_collection_name)
    video_chatter_summaries_collection = mongo_database.get_collection(video_chatter_summaries_collection_name)


    student_usernames = list(thread_collection.distinct("_student_username"))


    number_of_students = len(student_usernames)
    json_save_path =  Path(os.getenv("PATH_TO_COURSE_DROPBOX_FOLDER")) / "course_data" / "chatbot_data" / "video_chatter_summaries.json"
    json_save_path.parent.mkdir(parents=True, exist_ok=True)

    with get_openai_callback() as cb:
        random.shuffle(student_usernames)

        print(f"Student usernames: {'/n'.join(student_usernames)}")
        for student_iterator, student_username in enumerate(student_usernames):
            student_threads_in_designated_channel = [thread for thread in
                                                     thread_collection.find({'_student_username': student_username,
                                                                             "channel": designated_channel_name})]

            if len(student_threads_in_designated_channel) == 0:
                print(f"-----------------------------------------------------------------------------\n"
                      f"Student - ({student_username}) has no threads in {designated_channel_name}.\n"
                      f"-----------------------------------------------------------------------------\n")
                continue

            print(f"-----------------------------------------------------------------------------\n"
                  f"Generating VideoChatter for {student_username}\n"
                  f"Student#{student_iterator + 1} of {number_of_students}\n"
                  f"This student has e {len(student_threads_in_designated_channel)} threads in {designated_channel_name}.\n"
                  f"-----------------------------------------------------------------------------\n")

            for thread_number, thread_entry in enumerate(student_threads_in_designated_channel):

                student_discord_username = thread_entry["_student_username"]
                student_name = thread_entry["_student_name"]
                student_initials = "".join([name[0].upper() for name in student_name.split(" ")])
                thread_channel_name = thread_entry["channel"]
                server_name = thread_entry["server_name"]

                student_mongo_query = {
                    "_student_name": student_name,
                    "_student_username": student_discord_username,
                    "channel": thread_channel_name,
                    "server_name": server_name,
                }

                thread_as_big_string = "\n".join(thread_entry["thread_as_list_of_strings"])
                await mongo_database.upsert(collection=video_chatter_summaries_collection_name,
                                      query=student_mongo_query,
                                      data={"$set": {"threads": student_threads_in_designated_channel,
                                                     "thread_creation_time": thread_entry["created_at"],
                                                     },
                                            "$addToSet": {"thread_as_big_string": thread_as_big_string,}
                                            },
                                      )

                try:
                    video_chatter_summary_entry = video_chatter_summaries_collection.find_one(
                        {"_student_name": student_name})
                    current_video_chatter_summary = video_chatter_summary_entry.get("video_chatter_summary", "")
                    first_entry = False
                    if "video_chatter_summary" in video_chatter_summary_entry:
                        current_summary_created_at = video_chatter_summary_entry["video_chatter_summary"]["summary_creation_time"]
                        current_video_chatter_summary_created_at_datetime = datetime.strptime(
                            current_summary_created_at, '%Y-%m-%dT%H:%M:%S.%f')
                    else:
                        first_entry = True


                except Exception as e:
                    raise e

                video_chatter_summary_builder = VideoChatterSummaryBuilder(
                    student_name=student_name,
                    student_discord_username=student_discord_username,
                    current_summary=current_video_chatter_summary, )

                thread_summary = thread_entry['summary']['summary']
                thread_summary_created_at = thread_entry['summary']['created_at']
                thread_summary_created_at_datetime = datetime.strptime(thread_summary_created_at,
                                                                       '%Y-%m-%dT%H:%M:%S.%f')

                if not overwrite and not first_entry:
                    timedelta = thread_summary_created_at_datetime - current_video_chatter_summary_created_at_datetime
                    if timedelta.total_seconds() < 0:
                        print(
                            "Skipping thread because it is older than the current summary (i.e. it has already been incorporated into the summary)")
                        continue

                print(
                    f"---------Incorporating Thread#{thread_number + 1}-of-{len(student_threads_in_designated_channel)}-------------\n")
                print(f"Updating student summary based on thread with summary:\n {thread_summary}\n")

                print(f"Current student summary (before update):\n{current_video_chatter_summary}\n")

                updated_video_chatter_summary = await video_chatter_summary_builder.update_video_chatter_summary_based_on_new_conversation(
                    student_initials=student_initials,
                    current_schematized_summary=current_video_chatter_summary,
                    new_conversation_summary=thread_summary, )

                print(f"Updated summary (after update):\n{updated_video_chatter_summary}\n\n---\n\n")
                print(f"OpenAI API callback:\n {cb}\n")

                mongo_database.upsert(collection=video_chatter_summaries_collection_name,
                                      query=student_mongo_query,
                                      data={"$set": {"video_chatter_summary": {"thread_summary": thread_summary,
                                                                                "summary": updated_video_chatter_summary,
                                                                               "summary_creation_time": datetime.now().isoformat(),
                                                                               "model": video_chatter_summary_builder.llm_model}}},
                                      )
                if video_chatter_summary_entry is not None:
                    if "video_chatter_summary" in video_chatter_summary_entry:
                        mongo_database.upsert(collection=video_chatter_summaries_collection_name,
                                              query={"_student_username": student_username},
                                              data={"$push": {"previous_summaries": video_chatter_summary_entry[
                                                  "video_chatter_summary"]}}
                                              )
            mongo_database.save_json(collection_name=video_chatter_summaries_collection_name,
                                 save_path = json_save_path
                                 )




        mongo_database.save_json(collection_name=video_chatter_summaries_collection_name,
                             save_path = json_save_path
                             )


if __name__ == '__main__':
    server_name = "Neural Control of Real World Human Movement 2023 Summer1"
    thread_collection_name = get_thread_backups_collection_name(server_name=server_name)
    asyncio.run(generate_video_chatter_summaries(mongo_database=MongoDatabaseManager(),
                                                 thread_collection_name=thread_collection_name,
                                                 designated_channel_name="video-chatter-bot",
                                                 video_chatter_summaries_collection_name=VIDEO_CHATTER_SUMMARIES_COLLECTION_NAME,
                                                 use_anthropic=False,
                                                 overwrite=True,))
