import json
import os
import random
import string
import sys
import time
from io import BytesIO

import aiohttp
import art
import disnake
import ffmpeg
import openai
import validators
from disnake.ext import commands, tasks
from PIL import Image
from pytube import YouTube


class GeneralPurpose(commands.Cog):

    time_interval = random.randint(5, 15)

    cwd = os.getcwd()
    sys.path.append(f'{cwd}..')
    from config import ame_token, bg_key, open_ai

    from setup import ame_endpoints, horoscope, speech_bubble
    Templates = commands.option_enum(ame_endpoints)
    Horoscope = commands.option_enum(horoscope)
    movie_clips = json.load(open(f'db/movies_db.json', encoding='utf8'))
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\ AppleWebKit/537.36 (KHTML, like Gecko) \ Chrome/58.0.3029.110 Safari/537.36'}

    def __init__(self, bot):
        self.bot = bot
        openai.api_key = self.open_ai
        # self.test1.start()

    def trim_video(self, in_file: str, out_file: str, start: int, end: int):
        if os.path.exists(out_file):
            os.remove(out_file)

        in_file_probe_result = ffmpeg.probe(in_file)
        in_file_duration = in_file_probe_result.get(
            "format", {}).get("duration", None)
        print(in_file_duration)

        input_stream = ffmpeg.input(in_file)

        pts = "PTS-STARTPTS"
        video = input_stream.trim(start=start, end=end).setpts(pts)
        audio = (input_stream
                .filter_("atrim", start=start, end=end)
                .filter_("asetpts", pts))
        video_and_audio = ffmpeg.concat(video, audio, v=1, a=1)
        output = ffmpeg.output(video_and_audio, out_file, format="mp4")
        output.run()

        out_file_probe_result = ffmpeg.probe(out_file)
        out_file_duration = out_file_probe_result.get(
            "format", {}).get("duration", None)
        print(out_file_duration)


    def compress_video(self, video_full_path, size_upper_bound, two_pass=True, filename_suffix='cps_'):
        """
        Compress video file to max-supported size.
        :param video_full_path: the video you want to compress.
        :param size_upper_bound: Max video size in KB.
        :param two_pass: Set to True to enable two-pass calculation.
        :param filename_suffix: Add a suffix for new video.
        :return: out_put_name or error
        """
        filename, extension = os.path.splitext(video_full_path)
        print(filename)
        extension = '.mp4'
        output_file_name = filename + filename_suffix + extension

        # Adjust them to meet your minimum requirements (in bps), or maybe this function will refuse your video!
        total_bitrate_lower_bound = 11000
        min_audio_bitrate = 32000
        max_audio_bitrate = 256000
        min_video_bitrate = 100000

        try:
            # Bitrate reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
            probe = ffmpeg.probe(video_full_path)
            # Video duration, in s.
            duration = float(probe['format']['duration'])
            # Audio bitrate, in bps.
            audio_bitrate = float(next(
                (s for s in probe['streams'] if s['codec_type'] == 'audio'), None)['bit_rate'])
            # Target total bitrate, in bps.
            target_total_bitrate = (
                size_upper_bound * 1024 * 8) / (1.073741824 * duration)
            if target_total_bitrate < total_bitrate_lower_bound:
                print('Bitrate is extremely low! Stop compress!')
                return False

            # Best min size, in kB.
            best_min_size = (min_audio_bitrate + min_video_bitrate) * \
                (1.073741824 * duration) / (8 * 1024)
            if size_upper_bound < best_min_size:
                print('Quality not good! Recommended minimum size:',
                    '{:,}'.format(int(best_min_size)), 'KB.')
                # return False

            # Target audio bitrate, in bps.
            audio_bitrate = audio_bitrate

            # target audio bitrate, in bps
            if 10 * audio_bitrate > target_total_bitrate:
                audio_bitrate = target_total_bitrate / 10
                if audio_bitrate < min_audio_bitrate < target_total_bitrate:
                    audio_bitrate = min_audio_bitrate
                elif audio_bitrate > max_audio_bitrate:
                    audio_bitrate = max_audio_bitrate

            # Target video bitrate, in bps.
            video_bitrate = target_total_bitrate - audio_bitrate
            if video_bitrate < 1000:
                print('Bitrate {} is extremely low! Stop compress.'.format(video_bitrate))
                return False

            i = ffmpeg.input(video_full_path)
            if two_pass:
                ffmpeg.output(i, os.devnull,
                            **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4'}
                            ).overwrite_output().run()
                ffmpeg.output(i, output_file_name,
                            **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}
                            ).overwrite_output().run()
            else:
                ffmpeg.output(i, output_file_name,
                            **{'c:v': 'libx264', 'b:v': video_bitrate, 'c:a': 'aac', 'b:a': audio_bitrate}
                            ).overwrite_output().run()

            if os.path.getsize(output_file_name) <= size_upper_bound * 1024:
                return output_file_name
            elif os.path.getsize(output_file_name) < os.path.getsize(video_full_path):  # Do it again
                return self.compress_video(output_file_name, size_upper_bound)
            else:
                return False
        except FileNotFoundError as e:
            print('You do not have ffmpeg installed!', e)
            print('You can install ffmpeg by reading https://github.com/kkroening/ffmpeg-python/issues/251')
            return False

    @commands.slash_command(name='avatar', description='Get the avatar of a user.')
    async def avatar(self, inter, *, user: disnake.Member = None) -> None:
        """Get the avatar of a user."""
        if user is None:
            user = inter.author
        await inter.send(user.avatar.url)
        
    
    @commands.slash_command(name='shorten-url', description='Shorten any URL')
    async def shorten(self, inter, url: str) -> None:
        
        if not validators.url(url):
            return await inter.response.send_message('Please provide a valid URL', ephemeral=True)

        base_url = 'https://gotiny.cc/api'
        headers = {'Accept': 'application/json'}
        params = {'input': url}

        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, json=params, headers=headers) as resp:
                if resp.status != 200:
                    return await inter.response.send_message('Something went wrong', ephemeral=True)
                data = await resp.json()
                res = 'https://gotiny.cc/' + data[0]['code']
                await inter.response.send_message(res)


    @commands.slash_command(name='meme-generator' ,description='Generate a meme with the available templates')
    async def meme_generator(self, inter, img_url: str, template: Templates) -> None:

        await inter.response.defer(with_message='Loading...', ephemeral=False)
        
        if not validators.url(img_url):
            return await inter.response.send_message('Please provide a valid URL', ephemeral=True)
        
        base_url = "https://v1.api.amethyste.moe"
        headers = {'Authorization': f'Bearer {self.ame_token}'}
        data = {'url': img_url}

        async with aiohttp.ClientSession() as session:
            final_url = f'{base_url}/generate/{template}'
            async with session.post(final_url, data=data, headers=headers) as resp:
                if resp.status != 200:
                    return await inter.response.send_message('Something went wrong', ephemeral=True)
                data = await resp.content.read()

        bytes_io = BytesIO()
        image = Image.open(BytesIO(data))
        image.save(bytes_io, format='PNG')
        bytes_io.seek(0)
        dfile = disnake.File(bytes_io, filename=f'{template}.png')
        return await inter.followup.send(file=dfile)


    @commands.slash_command(description='Decode a QR code by providing a ')
    async def qr(self, inter, qr_url: str) -> None: 

        await inter.response.defer(with_message='Loading...', ephemeral=False)
        
        if not validators.url(qr_url):
            return await inter.followup.send('Please provide a valid URL', ephemeral=True)

        url = 'http://api.qrserver.com/v1/read-qr-code/?fileurl='
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{url}{qr_url}') as resp:
                if resp.status != 200:
                    return await inter.response.send_message('Something went wrong', ephemeral=True)
                data = await resp.json()

        return await inter.followup.send(data[0]['symbol'][0]['data'])


    @commands.slash_command(name='remove-background', description='Remove the background of an image')
    async def remove_background(self, inter, img_url: str) -> None:
        if not validators.url(img_url):
            return await inter.response.send_message('Please provide a valid URL', ephemeral=True)

        await inter.response.defer(with_message='Loading...', ephemeral=False)
        base_url = 'https://api.remove.bg/v1.0/removebg'
        headers = {'X-Api-Key': self.bg_key}
        data = {'image_url': img_url}

        async with aiohttp.ClientSession() as session:
            async with session.post(base_url, data=data, headers=headers) as resp:
                if resp.status != 200:
                    return await inter.response.send_message('Something went wrong', ephemeral=True)
                data = await resp.content.read()

        bytes_io = BytesIO()
        image = Image.open(BytesIO(data))
        image.save(bytes_io, format='PNG')
        bytes_io.seek(0)
        
        # send image
        await inter.followup.send(file=disnake.File(bytes_io, filename='bg_removed.png'))


    @commands.slash_command(name='movie-clip', description='Get a movie clip from the database')
    async def movie_clip(self, inter, movie: str) -> None:
        
        movie = movie.lower()
        flag = False
        for entry in self.movie_clips:
            if movie in entry.lower():
                movie = entry
                flag = True
                break
        
        if not flag:
            return await inter.response.send_message('No movie clip found', ephemeral=True)

        # get a random clip from the movie
        clip = random.choice(self.movie_clips[movie])
        return await inter.response.send_message(clip)


    @commands.slash_command(name='stoic', description='Get a stoic quote')
    async def stoic(self, inter) -> None:

        await inter.response.defer(with_message='Loading...', ephemeral=False)

        url = 'https://api.themotivate365.com/stoic-quote'
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return await inter.response.send_message('Something went wrong', ephemeral=True)
                data = await resp.json()

        title = data['data']['author']
        quote = data['data']['quote']
        embed = disnake.Embed(title=title, description=quote, color=0x00ff00)
        return await inter.followup.send(embed=embed)


    @tasks.loop(seconds=time_interval)
    async def test1(self) -> None:
        print('Waiting...')
        await self.bot.wait_until_ready()
        # send a message to the channel id = 953357475254505595
        rand_gid = random.choice(self.speech_bubble)
        self.time_interval = random.randint(5, 15)
        print(f'Time interval: {self.time_interval}')
        await self.bot.get_channel(953357475254505595).send(rand_gid)
        

    @commands.slash_command(name='channel-id', description='Get the channel ID')
    async def channel_id(self, inter) -> None:
        await inter.response.send_message(inter.channel.id)

    @commands.slash_command(name="phone-info", description="Get all information related to the phone nummber")
    async def phone_info(self, inter, phone_num: str, country_code: str = '1') -> None:
        # Getting rid of special chars
        phone_num = phone_num.replace('-', '').replace('(', '').replace(')', '').replace(' ', '').replace('+', '')
        phone_num = f'{country_code}{phone_num}'
        if not phone_num.isdigit():
            return await inter.response.send_message('Please provide a valid phone number', ephemeral=True)

        await inter.response.defer(with_message='Loading...', ephemeral=False)

        url = "https://phonevalidation.abstractapi.com/v1/?api_key=42253fea4b5645c9a45713308d620752&phone={phone}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url.format(phone=phone_num)) as resp:
                if resp.status != 200:
                    return await inter.response.send_message('Something went wrong', ephemeral=True)
                data = await resp.json()

        title = f'Phone: {data["phone"]}'
        valid = f'Validity: {data["valid"]}'
        international_format = f'International Format: {data["format"]["international"]}'
        local_format = f'Local Format: {data["format"]["local"]}'
        country = f'Country: {data["location"]}'
        carrier = f'Carrier: {data["carrier"]}'
        num_type = f'Number Type: {data["type"]}'

        description = [valid, international_format, local_format, country, carrier, num_type]
        embed = disnake.Embed(title=title, description='\n'.join(description), color=0x00ff00)

        return await inter.followup.send(embed=embed)

    
    @commands.slash_command(name='text-ascii', description='Get ascii art!')
    async def text_ascii(self, inter, text: str, font: str = 'block') -> None: 
        if len(text) > 20:
            return await inter.response.send_message('Please provide a text with less than 20 characters', ephemeral=True)

        if not text:
            return await inter.response.send_message('Please provide a text', ephemeral=True)

        try:
            ascii_text = art.text2art(text, font=font)
            
            if len(ascii_text) > 2000:
                # convert to text file as BytesIO
                bytes_io = BytesIO()
                bytes_io.write(ascii_text.encode())
                bytes_io.seek(0)
                file = disnake.File(bytes_io, filename='ascii_txt.txt')
                return await inter.response.send_message('Your message surpassed Discord 2000 character limit so we converted it to txt :)\n' ,file=file)

            return await inter.response.send_message(f'```{ascii_text}```')

        except art.artError:
            font = 'default'
            ascii_text = art.text2art(text, font=font)
            
            if len(ascii_text) > 2000:
                # convert to text file as BytesIO
                bytes_io = BytesIO()
                bytes_io.write(ascii_text.encode())
                bytes_io.seek(0)
                file = disnake.File(bytes_io, filename='ascii_txt.txt')
                return await inter.response.send_message('Your message surpassed Discord 2000 character limit so we converted it to txt :)\n' ,file=file)

            return await inter.response.send_message('The font you provided is not valid, using default font\n{ascii_art}', ephemeral=True)


    @commands.slash_command(name='horoscope', description='Get your horoscope fortune for the day.')
    async def horoscope(self, inter, sign: Horoscope) -> None: 
        url = 'https://aztro.sameerkumar.website/'
        params = (
        ('sign', sign),
        ('day', 'today'),
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params) as resp:
                if resp.status != 200:
                    return await inter.response.send_message('Something went wrong', ephemeral=True)
                data = await resp.json()

        title = f'Horoscope for {sign.capitalize()} -> {data["current_date"]}'
        description = [f"Today's fortune: {data['description']}"]
        embed = disnake.Embed(title=title, description='\n'.join(description), color=0x00ff00)

        return await inter.response.send_message(embed=embed)


    @commands.slash_command(name='random-person', description='Randomly generate the face of a person')
    async def random_person(self, inter) -> None:
        url = 'https://thispersondoesnotexist.com/image'
        await inter.response.defer(with_message='Loading...', ephemeral=False)

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return await inter.followup.send('Something went wrong', ephemeral=True)
                data = await resp.read()

        image = BytesIO(data)
        image.seek(0)
        file = disnake.File(image, filename='random_person.png')

        return await inter.followup.send(file=file, ephemeral=False)

    
    @commands.slash_command(name='youtube-embed', description='Embed a youtube video')
    async def youtube_embed(self, inter, url: str) -> None: 

        if not validators.url(url) or not 'youtube.com' in url:
            return await inter.response.send_message('Please provide a valid url', ephemeral=True)
        
        # if start or end:
        #     # If not numeric, ValueError will be raised, check if start and end are numeric
        #     if isinstance(start, str) and isinstance(end, str) and start.isnumeric() and end.isnumeric():
        #         # start = int(start)
        #         # end = int(end)
        #         pass
        #     else:
        #         return await inter.response.send_message('Please provide a valid start and end time', ephemeral=True)
            
        yt = YouTube(url)
        # length = yt.length

        await inter.response.defer(with_message='Loading...', ephemeral=False)

        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()

        if not stream:
            return await inter.followup.send('No mime type found for your video.', ephemeral=True)

        print("Downloading video...")
        stream.download(filename='youtube.mp4', output_path='db/')
        print("Finished download!")
        
        # vid_abs_path = os.path.abspath('db/youtube.mp4')
        # if end and end <= length:
        #     print("Trimming video...")
        #     self.trim_video(vid_abs_path, 'db/trim_vid.mp4', start, end)
        #     print("Finished trimming!")

        size = os.path.getsize('db/youtube.mp4')

        # check if size is larger than 8mb and less than 50mb
        if size > 8388608 and size < 52428800:
            await inter.followup.send('Your video is too large to send normally, compressing your video...', ephemeral=True)
            self.compress_video('db/youtube.mp4', 8 * 1000)
            file = disnake.File('db/youtubecps_.mp4', filename='youtubecps_.mp4')

            size = os.path.getsize('db/youtube.mp4')
            if size > 52428800:
                return await inter.followup.send(file=file, ephemeral=False)
            else:
                return await inter.followup.send('Your video is too large to send normally, please try a different video.', ephemeral=True)

        # file = disnake.File('db/youtube.mp4', filename=f'{stream.title}.mp4') if not end else disnake.File('db/trim_vid.mp4', filename=f'{stream.title}.mp4')
        file = disnake.File('db/youtube.mp4', filename=f'{stream.title}.mp4')
        return await inter.followup.send(file=file, ephemeral=False)


    @commands.slash_command(name='generate-image', description='Generate an image with a prompt')
    async def generate_image(self, inter, prompt: str = '') -> None:
        
        if not prompt:
            prompt = 'a white siamese cat'
            await inter.response.defer(with_message='No prompt provided, using default prompt\nLoading...', ephemeral=False)

        else:
            await inter.response.defer(with_message='Loading...', ephemeral=False)

        # we need to wait for the response before we can send the message
        response = openai.Image.create(prompt=prompt, n=1, size='1024x1024')
        # wait until be get the response url
        while not response:
            time.sleep(0.3)
            print('waiting for response url')

        return await inter.followup.send(response['data'][0]['url'], ephemeral=False)
        

    @commands.slash_command(name='youtube-thumbnail', description='Get the thumbnail of a youtube video')
    async def youtube_thumbnail(self, inter, url: str) -> None:
        if not validators.url(url) or not 'youtube.com' in url:
            return await inter.response.send_message('Please provide a valid url', ephemeral=True)

        yt = YouTube(url)
        thumbnail = yt.thumbnail_url

        if not thumbnail:
            return await inter.response.send_message('No thumbnail found', ephemeral=True)

        return await inter.response.send_message(thumbnail, ephemeral=False)


    # @commands.slash_command(name='weather', description='Get the live weather of a city')
    # async def weather(self, inter, city: str) -> None:
    #     await inter.response.defer(with_message='Loading...', ephemeral=False)
    #     title = f'Weather in {city}'
    #     city = city.replace(' ', '+')
    #     url = f'https://www.google.com/search?q={city}\&oq={city}\&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=\chrome&ie=UTF-8'

    #     async with aiohttp.ClientSession() as session:
    #         async with session.get(url, headers=self.headers) as resp:
    #             if resp.status != 200:
    #                 return await inter.followup.send('Something went wrong', ephemeral=True)
    #             data = await resp.text()

    #     soup = BeautifulSoup(data, 'html.parser')
    #     location = soup.select('#wob_loc').getText().strip()
    #     time = soup.select('#wob_dts').getText().strip()
    #     temp = soup.select('#wob_tm').getText().strip()
    #     desc = soup.select('#wob_dc').getText().strip()
    #     humidity = soup.select('#wob_hm').getText().strip()
    #     description = [location, time, temp, desc, humidity]
    #     embed = disnake.Embed(title=title, description='\n'.join(description), color=0x00ff00)

    #     return await inter.followup.send(embed=embed)


    # @commands.slash_command(name='ascii-art', description='Produce ascii art from an image')
    # async def ascii_art(self, inter, image: disnake.Attachment) -> None:
    #     await inter.response.defer(with_message='Loading...', ephemeral=False)

    #     if not image:
    #         return await inter.followup.send('Please provide an image', ephemeral=True)

    #     # Store image in BytesIO
    #     image = BytesIO(await image.read())
    #     kit.image_to_ascii_art(image, 'ascii')
    #     return await inter.followup.send(file=disnake.File('ascii.txt'))


def setup(bot):
    bot.add_cog(GeneralPurpose(bot))