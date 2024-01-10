"""This file is our own customised toolset for the Agent"""
import sys
sys.path.append('/home/pi/Desktop/GPTSpeaker')


from langchain.tools import BaseTool
from iodevices.TemperatureHumiditySensor import TemperatureHumiditySensor
from iodevices.rainning_check import RainSensor
from iodevices.RGB import RGBLED
import Adafruit_DHT
import time
class IndoorTemperatureHumidity(BaseTool):
    name = "IndoorTemperatureHumidity"
    description = "当你需要知道室内的温度和湿度时使用该工具，你需要传入的action_input为空"
    def _run(self,query: str) -> str:
        """return Temperature and Humidity"""
        temperatureHumiditySensor = TemperatureHumiditySensor(Adafruit_DHT.DHT11, 4)
        temperature,humidity = temperatureHumiditySensor.read_temperature_and_humidity()
        
        # Test case
        # temperature,humidity = 23,55

        #Check for correct return of temperature and humidity
        try:
            self.check_not_none(temperature)
            self.check_not_none(humidity)
        except ValueError as e:
            print(e)
        return '''当前室内的温度为:{}°C  当前室内的湿度为:{}\%'''.format(temperature,humidity)

    def check_not_none(self,a):
        if a is not None:
            return True
        else:
            raise ValueError("Temperature and humidity cannot be None!")
        
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("IndoorTemperatureHumidity does not support async")


class CheckRaining(BaseTool):
    name = "CheckRaining"
    description = "当你被要求使用雨滴传感器监测是否下雨时使用该工具"
    def _run(self, query:str) -> str:
        """Return to Whether it's raining or not """
        
        # TODO No parameters for the pins have been passed in yet, this part is not complete
        rainSensor = RainSensor(13)
        isRaining = rainSensor.is_raining()
        # isRaining = True
        # Check for correct return of isRaining
        try:
            self.check_not_none(isRaining)
        except ValueError as e:
            print(e)
        
        if isRaining:
            return "现在正在下雨"
        else:
            return "现在没有在下雨"

    def check_not_none(self,a):
        if a is not None:
            return True
        else:
            raise ValueError("isRaining cannot be None!")
        
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CheckRaining does not support async")


class TurnOnLight(BaseTool):
    name = "TurnOnLight"
    description = "该工具被用来打开室内的灯,其action_input为空"
    def _run(self, query:str) -> str:
        red_pin = 18
        green_pin = 23
        blue_pin = 24

        # 创建RGBLED对象
        led = RGBLED(red_pin, green_pin, blue_pin)
        # 循环改变LED灯的颜色
        colors = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF"]
        for color in colors:
            print("Setting color: " + color)
            led.set_color_hex(color)   # 设置LED灯的颜色
            time.sleep(1)   # 延时1秒

        # 清理GPIO设置并释放资源
        led.cleanup()
        return "灯已打开"
    async def _arun(self, query:str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("TurnOnLight does not support async")