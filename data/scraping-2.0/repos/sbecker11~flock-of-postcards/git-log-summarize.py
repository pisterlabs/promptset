from datetime import datetime, timedelta
from collections import defaultdict

import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

async def main() -> None:
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )

asyncio.run(main())



log = """
Mon Jan 1 13:52:15 2024 -0700 fixed monocolor issues, renamed ICON_TYPES and icon filenames
Tue Dec 26 15:24:40 2023 -0700 added month/years of experience for cardDivLineItem with target cardDiv
Sat Dec 23 23:52:16 2023 -0700 add back-icon, delete back-icon when cardDivLineItem references a bizcard
Fri Dec 22 13:45:04 2023 -0700 applyMonoColorToElement
Wed Dec 20 21:22:19 2023 -0700 added url and icons on newline when added to cardivs and in-line when added to cardDivLineItems with color of parent element
Wed Dec 20 17:42:37 2023 -0700 moved welcomeAlert to alerts.mjs, black and white icons added
Wed Dec 20 01:29:32 2023 -0700 diagnostics removed, styles changed, TagLinks now working correctly.
Tue Dec 19 00:50:06 2023 -0700 tagLink spans now visible in both cardDivs and cardDivLineItems
Mon Dec 18 17:10:38 2023 -0700 monoColor.mjs
Mon Dec 18 13:40:14 2023 -0700 added "Click here to get started" modal
Sun Dec 17 19:01:18 2023 -0700 can handle NUM_ANIMATION_FRAMES = 0
Sun Dec 17 18:50:33 2023 -0700 fixed StyleArray vs StyleProps
Sat Dec 16 14:36:52 2023 -0700 validate utils functions added
Fri Dec 15 18:24:00 2023 -0700 colors and greys icons added
Fri Dec 15 03:22:53 2023 -0700 added world and image icons, revised scroll into view
Thu Dec 14 18:21:50 2023 -0700 text-img-url tagLinks getting better
Thu Dec 14 00:52:23 2023 -0700 select first and select next working
Mon Dec 11 18:42:31 2023 -0700 email changed
Mon Dec 11 00:46:29 2023 -0700 cardivLineItem.innerHTML  = tagLink.text<br/>tagLink.url
Sun Dec 10 17:06:16 2023 -0700 refreshed convert-jobs-xslx-to-mjs.sh
Sun Dec 3 19:42:32 2023 -0700 Removed Select Skills button
Sun Dec 3 19:34:57 2023 -0700 added Select First button, resetting divStyleArray[z] to originalZ when negative, added flag to selectTheX and deselectTheX to select or deselect derivative element (divCard<->divCardLineItem (saving lots of code), only apply animations on non-lineItem elements, implementing custom scrollElementIntoView,
Sat Dec 2 22:45:25 2023 -0700 entire project converted to Node ES6 format using a package.json file and all .js files renamed to .mjs
Sat Dec 2 22:06:18 2023 -0700 created tests/test_utils and static_content/media/scale-wordpress-images.zip
Sat Dec 2 20:55:59 2023 -0700 added event listener to block click events while animation is in progress to ensure animation is handled without interruption. And introduced targetParallaxedDivStyleArray so div is restored to original position with parallax applied.
Sat Nov 25 22:45:47 2023 -0700 Added planned features
Sat Nov 25 22:42:54 2023 -0700 made selected style !important, added arrayHasNaNs checks to all divStyleArrays, simplified scrollElementIntoView, trying unsuccessfully to easeFocalPointToBullsEye when any div is selected.
Sat Nov 25 00:43:13 2023 -0700 Reverted attempts to call "selectAllBizCards" on window load
Fri Nov 24 09:22:35 2023 -0700 border widths and scrollIntoView adjustments
Thu Nov 23 22:02:33 2023 -0700 version in title
Thu Nov 23 21:43:23 2023 -0700 version-0.8 new graphics added
Thu Nov 23 20:40:47 2023 -0700 merge version-0.7 with master
Thu Nov 23 20:22:55 2023 -0700 Create common ancestor commit
Thu Nov 23 20:16:52 2023 -0700 track *.zip files using Git LFS
Thu Nov 23 20:05:56 2023 -0700 Features planned
Thu Nov 23 19:47:51 2023 -0700 Auto-computing timeline min-max years from jobs jobs file, interpolating CURRENT_DATE  in jobs file, default timeline year as avg of min-max years"""


# Split the log into lines
lines = log.strip().split("\n")

# Group lines by week
weeks = defaultdict(str)
for line in lines:
    
    date_str = " ".join(line.split(" ", 6)[:6])
    message = line.split(" ", 6)[6]
    date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y %z")
    formatted_date = date.strftime("%Y-%m-%d")

    # Subtract the day of the week from the date to get the previous Sunday
    sunday_datetime = (date - timedelta(days=date.weekday() + 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    sunday = sunday_datetime.strftime("%Y-%m-%d")
    if sunday not in  weeks:
        weeks[sunday] = ""
    sep = " | " if weeks[sunday] else ""
    weeks[sunday] += sep + message

# Concatenate messages for each week

# Print weekly messages
for week in sorted(weeks, reverse=True):
    print(f"{week}:\n{weeks[week]}")