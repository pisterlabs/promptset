import wave
import time
import sys
import os
import re
from threading import Thread

import pyaudio

from urllib import request as ur
from urllib import parse

import requests

import openai
import pyperclip

import lib.utils.util_stopwatch as sw

# from src.lib import util as du
import lib.utils.util as du

wav_file     = "vox-to-be-transcribed.wav"
docker_path  = "/var/io/{}"
local_path   = "/Volumes/projects/io/{}"
write_method = "file" # "file" or "flask"

GPT_4   = "gpt-4-0613"
GPT_3_5 = "gpt-3.5-turbo-0613"

class GenieClient:
    
    
    def __init__( self, calling_gui=None, startup_mode="transcribe_and_clean_prose",
                  prefix="multimodal text punctuation", copy_transx_to_clipboard=True, runtime_context="docker",
                  write_method="flask", debug=False, recording_timeout=30, stt_address="127.0.0.1:7999",
                  tts_address="127.0.0.1:5002", tts_output_path="/var/io/tts.wav"
                  ):
        
        self.debug = debug
        self.bar = "*" * 80
        
        self.CHUNK         = 4096
        self.FORMAT        = pyaudio.paInt16
        self.CHANNELS      = 1
        self.RATE          = 44100
        self.recording     = False
        self.playing       = False
        
        if debug: print( "runtime_context [{}]".format( runtime_context ) )
        if runtime_context == "docker":
            self.output_path = docker_path.format( wav_file )
        else:
            self.output_path = local_path.format( wav_file )
        if debug: print( "Setting runtime output_path to [{}]".format( self.output_path ) )

        self.startup_mode       = startup_mode
        self.prefix             = prefix
        self.project_root       = du.get_project_root()
        
        if debug: print( "         self.project_root [{}]".format( self.project_root ) )
        if debug: print( "translation-dictionary.map [{}]".format( self.project_root + "/src/conf/translation-dictionary.map" ) )
        
        self.punctuation        = du.get_file_as_dictionary( self.project_root + "/src/conf/translation-dictionary.map", lower_case=True )
        self.prompts_dict       = du.get_file_as_dictionary( self.project_root + "/src/conf/prompts.map", lower_case=False )
        self.modes_dict         = du.get_file_as_json(       self.project_root + "/src/conf/modes.json" )
        self.methods_dict       = self._get_titles_to_methods_dict()
        self.keys_dict          = self._get_titles_to_keys_dict()
        self.prompt_titles      = self._get_prompt_titles()
        self.default_mode_index = self._get_default_mode_index()
        self.calling_gui        = calling_gui
        self.recording_timeout  = recording_timeout
        self.runtime_context    = runtime_context
        self.write_method       = write_method
        self.input_path         = "/var/io/{}".format( wav_file )
        self.stt_address        = stt_address
        self.tts_address        = tts_address
        self.tts_wav_path       = tts_output_path
        self.py                 = pyaudio.PyAudio()
        
        self.sound_resources_dir = self.project_root + "/src/resources/sound/"
        
        # tracks what the recording & writing thread is doing
        self.finished_serializing_audio = False

        # Do we want to automatically stash the data of a transcription or processing of a transcription to the clipboard when we're done?
        self.copy_transx_to_clipboard   = copy_transx_to_clipboard
        
        self.finished_transcription     = None
 
    def get_titles( self ):
    
        titles = [ ]
        for key in self.modes_dict.keys():
            titles.append( self.modes_dict[ key ][ "title" ] )
    
        return titles

    def _get_titles_to_methods_dict( self ):
    
        title_to_methods_dict = { }
        for key in self.modes_dict.keys():
            title_to_methods_dict[ self.modes_dict[ key ][ "title" ] ] = self.modes_dict[ key ][ "method_name" ]
    
        return title_to_methods_dict

    def _get_titles_to_keys_dict( self ):

        title_to_keys_dict = { }
        for key in self.modes_dict.keys():
            title_to_keys_dict[ self.modes_dict[ key ][ "title" ] ] = key

        return title_to_keys_dict

    def _get_prompt_titles( self ):

        titles = [ key for key in self.prompts_dict.keys() ]
        titles.sort()

        return titles
    def _get_default_mode_index( self ):
    
        modes_list = list( self.modes_dict.keys() )
        
        if self.debug: print( "Default mode [{}]".format( self.startup_mode ) )
        if self.debug: print( "modes_list", modes_list )
        
        if self.startup_mode not in modes_list:
            
            print( "ERROR: Default mode [{}] not found in modes list [{}]".format( self.startup_mode, modes_list ) )
            print( "ERROR: Setting default mode to [{}]".format( modes_list[ 0 ] ) )
            self.startup_mode = modes_list[ 0 ]
            index = 0
        else:
            index = modes_list.index( self.startup_mode )
            if self.debug: print( "Default mode index [{}]".format( index ) )
        
        return index
    
    def process_by_mode_title( self, mode_title ):
    
        function_name = self.methods_dict[ mode_title ]
        print()
        if self.debug: print( "Calling [{}]...".format( function_name ) )
        transcription = getattr( self, function_name )()
        
        return transcription
            
    def is_recording( self ):
        
        return self.recording

    def stop_recording( self ):
    
        self.recording = False

    def is_playing( self ):
        
        return self.playing
    
    def stop_playing( self ):
    
        self.playing = False
        
    def _start_recording_timeout( self, seconds ):
    
        if self.debug: print( "Starting recording timeout thread, {} seconds limit...".format( seconds ) )
        thread = Thread( target=self._wait_to_stop_recording, args=[ seconds ] )
        thread.start()
    
    def _wait_to_stop_recording( self, max_recording_seconds=30 ):
        
        if self.debug: print( "Waiting to stop recording [{}] for [{}] seconds...".format( self.recording, max_recording_seconds ) )
        seconds = 0
        self.timer_running = True
        
        while seconds < max_recording_seconds and self.timer_running:
            time.sleep( 1 )
            if self.debug: print( "*", end="" )
            seconds += 1
        
        self.recording = False
        
    def stop_recording_timeout( self ):
    
        self.timer_running = False

    def start_recording( self ):

        if self.debug: print( "Recording...", end="" )
        
        # This was removed
        # self.play_recording()

        self.recording = True
        self.finished_serializing_audio = False
        self._start_recording_timeout( seconds=self.recording_timeout )

        frames_buffer = [ ]
        stream = self.py.open( format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK )
        while self.recording:
            data = stream.read( self.CHUNK )
            frames_buffer.append( data )
            if self.debug: print( ".", end="" )
            # only call if there's a GUI to update
            if self.calling_gui is not None: self.calling_gui.update()

        if self.debug:
            print( " Done!" )
            print( "Frames recorded [{}]".format( len( frames_buffer ) ) )
        stream.close()
    
        self.stop_recording_timeout()
        
        if self.write_method == "flask":
            
            if self.debug: print( "POST'ing audio to [{}]...".format( self.write_method ) )
            
            # Write to temp file
            temp_file = "/tmp/{}".format( wav_file )
            if self.debug: print( "Writing audio to temp file for POST'ing [{}]...".format( temp_file ), end="" )
            self._write_audio_file( temp_file, frames_buffer )
            if self.debug: print( " Done!" )
            
            # Post temp file to flask server
            files = [ ( "file", ( wav_file, open( temp_file, "rb" ), "multipart/form-data" ) ) ]
            url   = "http://{}/api/upload-and-transcribe-wav?prefix={}".format( self.stt_address, self.prefix )
            
            if self.debug: print( "POST'ing tempfile [{}] to [{}]...".format( temp_file, url ), end="" )
            response = requests.request( "POST", url, headers={ }, data={ }, files=files )
            if self.debug: print( " Done!" )

            # Delete temp file
            if self.debug: print( "Deleting temp file [{}]...".format( temp_file ), end="" )
            os.remove( temp_file )
            if self.debug: print( " Done!" )

            transcribed_text = response.text
            if self.debug: print( "Transcription returned [{}]".format( transcribed_text ) )
            return transcribed_text
            
        self.finished_serializing_audio = True
        
    def ask_chat_gpt_using_raw_prompt_and_content( self, prompt_and_content, model=GPT_4 ):
        
        openai.api_key = os.getenv( "FALSE_POSITIVE_API_KEY" )
        print( "Using FALSE_POSITIVE_API_KEY [{}]".format( os.getenv( "FALSE_POSITIVE_API_KEY" ) ) )
        
        print( "prompt_and_content [{}]".format( prompt_and_content ) )
        prompt  = prompt_and_content.split( "```" )[ 0 ].strip()
        content = prompt_and_content.split( "```" )[ 1 ].strip()
        
        if self.debug:
            print( " prompt [{}]".format( prompt ) )
            print( "content [{}]".format( content ) )
        
        response = openai.chat.completions.create(
            model=model,
            messages=[
                { "role": "system", "content": prompt },
                { "role": "user", "content": content }
            ],
            # From: https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api-a-few-tips-and-tricks-on-controlling-the-creativity-deterministic-output-of-prompt-responses/172683
            # Using creative writing values
            temperature=0.5,
            top_p=0.5,
            max_tokens=3400,
            # From: https://community.openai.com/t/difference-between-frequency-and-presence-penalties/2777/2
            frequency_penalty=0.5,
            presence_penalty=0.5
            # # Non-zero values break JSON formatting.
            # frequency_penalty=0.0,
            # presence_penalty=0.0
        )
        if self.debug: print( response )
        
        return response.choices[ 0 ].message.content.strip()
    
    def run_fine_tuning( self, prompt_and_content ):
        
        openai.api_key = os.getenv( "FALSE_POSITIVE_API_KEY" )
        # TODO: Sketch out fine tuning parameters in notebook and add here
        
    def ask_chat_gpt_text( self, query, preamble="What does this mean: ", model=GPT_4 ):

        openai.api_key = os.getenv( "FALSE_POSITIVE_API_KEY" )
        if self.debug: print( "Using FALSE_POSITIVE_API_KEY [{}]".format( os.getenv( "FALSE_POSITIVE_API_KEY" ) ) )
    
        timer = sw.Stopwatch()
        print( "Asking ChatGPT [{}]...".format( model ), end="" )
        response = openai.chat.completions.create(
            model=model,
            messages=[ { "role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. "
                                                      "Answer as concisely as possible." },
                       { "role": "system", "content": preamble },
                       { "role": "user", "content": query } ],
            temperature=0,
            max_tokens=600,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        timer.print( "Done!".format( model ), use_millis=True )
        if self.debug: print( response )
        
        return response.choices[ 0 ].message.content.strip()

    def ask_chat_gpt_code( self, query, preamble="Fix this source code", model=GPT_4 ):
    
        openai.api_key = os.getenv( "FALSE_POSITIVE_API_KEY" )
    
        response = openai.completions.create(
            model=model,
            prompt="{}\n\n###{}###".format( preamble, query ),
            temperature=0,
            max_tokens=600,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=[ "###" ]
        )
        print( response )
    
        return response.choices[ 0 ].message.content.strip()
    
    def get_tts_file( self, ip_and_port, tts_input, tts_output_path ):
    
        text = parse.quote_plus( tts_input )
    
        timer = sw.Stopwatch()
        url = "http://{ip_and_port}/api/tts?text={text}".format( ip_and_port=ip_and_port, text=text )
        print( "Converting text to speech [{}] and writing to temp file [{}]...".format( url, tts_output_path ), end="" )
        response = ur.urlopen( url ).read()
    
        with open( tts_output_path, "wb" ) as f:
            f.write( response )
            
        timer.print( "Done" )
    
    def play_file( self, tts_wav_path ):
    
        print( "Playing [{}]...".format( tts_wav_path ) )
        self.playing = True
        
        with wave.open( tts_wav_path, "rb" ) as wf:
            # Instantiate PyAudio and initialize PortAudio system resources (1)
            p = pyaudio.PyAudio()
        
            # Open stream (2)
            stream = p.open( format=p.get_format_from_width( wf.getsampwidth() ),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True
                            )
        
            # Play samples from the wave file (3)
            while len( data := wf.readframes( self.CHUNK ) ):  # Requires Python 3.8+ for :=
                
                stream.write( data )
                
                # Give the GUI a moment to breathe
                if self.calling_gui is not None: self.calling_gui.main.update()
                # Allow the GUI or caller to stop playback.
                if not self.playing: break
        
            # Close stream (4)
            stream.close()
        
            # Release PortAudio system resources (5)
            p.terminate()
    
    def do_gpt_by_voice( self ):
    
        transcribed_text = self.start_recording()

        # transcribed_text = self._get_transcription( self.stt_address, self.serialized_audio_path )

        self.play_working()
        gpt_response = self.ask_chat_gpt_text( transcribed_text )

        # Copy the question and answer to the clipboard and print to the console.
        q_and_a = "Q: {}\n\nA: {}".format( transcribed_text, gpt_response )
        self.copy_to_clipboard( q_and_a )
        
        self.get_tts_file( self.tts_address, gpt_response, self.tts_wav_path )
        
        self.play_file( self.tts_wav_path )

    def play_oops( self ):
    
        self.play_file( self.sound_resources_dir + "oops.wav" )

    def play_working( self ):
        
        self.play_file( self.sound_resources_dir + "working.wav" )

    
    def do_gpt_from_clipboard( self ):
    
        self.play_working()
        
        clipboard_text = self.get_from_clipboard()

        gpt_response = self.ask_chat_gpt_text( clipboard_text )

        # Copy the question and answer to the clipboard and print to the console.
        q_and_a = "Q: {}\n\nA: {}".format( clipboard_text, gpt_response )
        self.copy_to_clipboard( q_and_a )
        # gpt_response = "Oops, I didn't understand that."
        
        self.play_working()
        self.get_tts_file( self.tts_address, gpt_response, self.tts_wav_path )
    
        self.play_file( self.tts_wav_path )

    def do_read_to_me( self ):
    
        clipboard_text = self.get_from_clipboard()
    
        self.play_working()
        self.get_tts_file( self.tts_address, clipboard_text, self.tts_wav_path )
    
        self.play_file( self.tts_wav_path )

    def do_transcribe_and_clean_prose( self ):
    
        self.prefix = "multimodal text punctuation"
        raw_text = self.do_transcription()
    
        preamble = "You are an expert copy editor. Clean up the following text, including using proper capitalization, " \
                   "contractions, grammar, and translation of punctuation mark words into single characters. Format " \
                   "output as formal technical prose."

        gpt_response = self.ask_chat_gpt_text( raw_text, preamble=preamble )

        self.copy_to_clipboard( gpt_response )

    def do_transcribe_and_clean_python( self ):
    
        self.prefix = "multimodal python punctuation"
        python_code = self.do_transcription( copy_to_clipboard=self.copy_transx_to_clipboard )
        if self.copy_transx_to_clipboard: self.copy_to_clipboard( python_code )

        return python_code
        
    def do_gpt_code_explanation_from_clipboard( self ):
        
        query = self.get_from_clipboard()
        preamble = "Explain the following code or error message in natural language. Treat this text as an error only if you see the word error."
        gpt_response = self.ask_chat_gpt_text( query, preamble=preamble )

        # self._copy_to_clipboard( gpt_response )

        self.get_tts_file( self.tts_address, gpt_response, self.tts_wav_path )

        self.play_file( self.tts_wav_path )

    def do_gpt_prose_explanation_from_clipboard( self ):

        query = self.get_from_clipboard()
        preamble = "Explain the following text in natural language: "
        gpt_response = self.ask_chat_gpt_text( query, preamble=preamble )

        self.copy_to_clipboard( gpt_response )

        self.get_tts_file( self.tts_address, gpt_response, self.tts_wav_path )

        self.play_file( self.tts_wav_path )
    def munge_code( self, code ):
    
        if self.debug: print( "Before punctuation translation: \n\n{}".format( code ), end="\n\n" )
        
        # Remove "space, ", commas, and periods.
        code = re.sub( r'space, |[,]', '', code.lower() )

        # Translate punctuation mark words into single characters.
        for key, value in self.punctuation.items():
            code = code.replace( key, value )

        # Remove extra spaces.
        code = code.replace( " _ ", "_" )
        code = code.replace( " ,", "," )
        code = code.replace( "self . ", "self." )
        code = code.replace( " . ", "." )
        code = code.replace( "[ { } ]", "[{}]" )
        code = code.replace( " ( )", "()" )
        code = code.replace( ") :", "):" )
        code = code.replace( " ( ", "( " )
        # code = code.replace( " ) ", " ) " )
        
        # Remove extra spaces
        code = ' '.join( code.split() )

        if self.debug: print( "After punctuation substitution: \n\n{}".format( code ), end="\n\n" )
        
        return code

    def munge_prose( self, prose ):

        print( "munge_prose() Before punctuation translation: \n\n{}".format( prose ), end="\n\n" )

        # Remove commas, and periods.
        prose = re.sub( r'[,.]', '', prose.lower() )

        # Translate punctuation mark words into single characters.
        for key, value in self.punctuation.items():
            prose = prose.replace( key, value )

        # Remove extra spaces.
        prose = prose.replace( "[ ", "[" )
        prose = prose.replace( " ]", "]" )
        prose = prose.replace( " )", ")" )
        prose = prose.replace( "( ", "(" )
        prose = prose.replace( " .", "." )
        prose = prose.replace( " ,", "," )
        prose = prose.replace( " ?", "?" )
        # prose = prose.replace( "??", "?" )
        prose = prose.replace( " !", "!" )
        # prose = prose.replace( "!!", "!" )
        prose = prose.replace( " :", ":" )
        prose = prose.replace( " ;", ";" )
        prose = prose.replace( ' "', '"' )

        prose = ' '.join( prose.split() )

        print( "munge_prose() After punctuation substitution: \n\n{}".format( prose ), end="\n\n" )

        return prose
    
    def do_transcription( self, copy_to_clipboard=True ):
    
        transcribed_text = self.start_recording()
        
        if copy_to_clipboard: self.copy_to_clipboard( transcribed_text )
        
        return transcribed_text

    # def do_vox_edit_of_prose_prompt( self ):
    #
    #     print( "do_vox_edit_of_prose_prompt() called..." )
    #     transcript = self.do_transcription( copy_to_clipboard=False )
    #     transcript = self.munge_prose( transcript )
    #     self.copy_to_clipboard( transcript )
    #     self.calling_gui.txt_response.insert( "1.0", transcript )
    #     self.calling_gui.last_text_with_focus.focus_force()
    #
    # def do_process_prose_prompt( self ):
    #
    #     print( "do_process_prose_prompt() called..." )
    #     response = "\n\n" + self.ask_chat_gpt_text( self.calling_gui.txt_content.get('1.0', 'end'), preamble=self.calling_gui.txt_prompt.get('1.0', 'end') ) + "\n\n"
    #     hr = "-" * 50
    #     hr = hr + "\n\n"
    #     self.copy_to_clipboard( response )
    #     self.calling_gui.txt_response.insert( "1.0", hr )
    #     self.calling_gui.txt_response.insert( "1.0", response )
    #     self.calling_gui.last_text_with_focus.focus_force()

    def copy_to_clipboard( self, text ):
        
        pyperclip.copy( text )
        # print( "Copied to clipboard: \n\n{}".format( text ), end="\n\n" )
        
    def get_from_clipboard( self ):
        
        return pyperclip.paste()
    
    def _write_audio_file( self, path, frames_buffer ):
    
        wf = wave.open( path, "wb" )
        wf.setnchannels( self.CHANNELS )
        wf.setsampwidth( self.py.get_sample_size( self.FORMAT ) )
        wf.setframerate( self.RATE )
        wf.writeframes( b"".join( frames_buffer ) )
        wf.close()

if __name__ == "__main__":
    
    # print( "Starting GenieClient in [{}]...".format( os.getcwd() ) )
    cli_args = du.get_name_value_pairs( sys.argv )

    # runtime_context   = cli_args.get( "runtime_context", "docker" )
    # write_method      = cli_args.get( "write_method", "flask" )
    # startup_mode      = cli_args.get( "startup_mode", "transcribe_and_clean_python" )
    startup_mode      = cli_args.get( "startup_mode", "transcribe_and_clean_prose" )
    recording_timeout = int( cli_args.get( "recording_timeout", 5 ) )
    
    print( "     startup_mode: [{}]".format( startup_mode ) )
    # print( "  runtime_context: [{}]".format( runtime_context ) )
    # print( "     write_method: [{}]".format( write_method ) )
    print( "recording_timeout: [{}]".format( recording_timeout ) )
    
    gc = GenieClient( startup_mode=startup_mode, debug=True, recording_timeout=recording_timeout )
    # transcription = gc.do_transcribe_and_clean_python()
    transcription = gc.do_transcription()
    print( f"Transcription [{transcription}]" )

    # gc.do_gpt_by_voice()
    # code = gc.munge_code( "Deaf key underscore event open parenthesis self comma event close parenthesis colon new line new line")
    # print( code )


    # transcription = gc.do_transcription()
    # print( transcription )
    # gc.do_gpt_by_voice()
    # gc.do_gpt_from_clipboard()
    # gc.do_transcribe_and_clean()
    # gc.play_recording()
    
    # preamble = "Clean up the following raw text transcription created by whisper.ai. Correct spelling and format " \
    #           "output as Python source code w/o any capitalization. Source code must compile" \
    #
    # query = "Let me stop you right there, Colin. First, comma. You'd better stop shouting so much comma. Okay, question mark. By the way comma, what's it like to open bracket almost exclusively, close bracket, use your voice comma instead of your hands. question mark. "
    # munged_prose = gc.munge_prose( query )
    # print( munged_prose )
    # preamble = "You are an expert copy editor. Clean up the following text, including using proper capitalization, " \
    #            "contractions, grammar, and translation of punctuation mark words into single characters. Format " \
    #            "output as formal technical prose."
    #
    # for i in range( 1 ):
    #     gpt_response = gc.ask_chat_gpt_text( munged_prose, preamble=preamble )
    #     print( gpt_response )
    
    #
    # print( "1) Before punctuation translation: \n\n{}".format( query ), end="\n\n" )
    # # query = query.lower().replace( ",", "" ).replace( "space, ", "" )
    # query = query.lower().replace( ",", "" ).replace( ".", "" ).replace( "space, ", "" )
    # print( "2) Before punctuation translation: \n\n{}".format( query ), end="\n\n" )
    #
    # for key, value in gc.punctuation.items():
    #     query = query.replace( key, value )
    #
    # query = query.replace( " _ ", "_" ).replace( " ,", "," ).replace( ") :", "):" ).replace( "self . ", "self." ).replace( " . ", "." ).replace( " ( )", "()" ).replace( "[ { } ]", "[{}]" ).replace( " ( ", "( " )
    # query = ' '.join( query.split() )
    #
    #
    # print( "After punctuation translation: \n\n{}".format( query ), end="\n\n" )
    # query_list = query.split( " " )
    # new_query_list = []
    # for word in query_list:
    #
    #     print( "word: [{}]".format( word ), end="" )
    #     word = gc.punctuation.get( word, word )
    #     print( "word: [{}]".format( word ) )
    #     new_query_list.append( word )
    #
    # query = " ".join( new_query_list )
    # print( "After punctuation substitution: \n\n{}".format( query ), end="\n\n" )

    
    # data = gc.ask_chat_gpt_code( query, preamble=preamble )
    # question _ embedding
    # print( "Results: \n\n{}".format( data ), end="\n\n" )
    #
    # gc.process_by_mode_title( gc.modes_dict[ default_mode ][ "title" ] )
    # gc.start_recording()
    # gc.wait_to_finish_audio_serialization( gc, max_wait_seconds=10 )
    
    # Not much, you know, just two or three hours a week.