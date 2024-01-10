import os
import requests
import datetime

from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from langchain.agents import tool

load_dotenv(find_dotenv())

base_location = os.getenv("BASE_LOCATION")


class TrainScheduleInput(BaseModel):
    """Input for Train Schedule Check"""
    station_name: str = Field(
        ..., description=f"Name of the station to check, if not provided always use {base_location}")


@tool(args_schema=TrainScheduleInput)
def get_schedule(station_name: str) -> str:
    """Get the train time table for a given station. it only this outputs the first and last time schedule and recent 1hours of schedule from now time"""

    def get_station_code(station_name: str):
        return "102"

    def get_day_code(weekday: int):
        # weekday: 0 ~ 4인 경우 8, 5인경우 7, 6인경우 9을 반환
        # 공휴일을 판별하는 로직 삽입 필요함.
        if weekday < 5:
            return "8"
        elif weekday == 5:
            return "7"
        elif weekday == 6:
            return "9"
        raise Exception("Invalid weekday")

    def run(station_name: str):
        now = datetime.datetime.now()

        start_time = now - datetime.timedelta(minutes=15)
        end_time = now + datetime.timedelta(hours=1)
        stinCd = get_station_code(station_name)
        dayCd = get_day_code(now.weekday())

        params = {
            "serviceKey": os.environ.get("TRAIN_PORTAL_API_KEY"),
            "dayCd": dayCd,
            "lnCd":  stinCd[0],
            "stinCd": stinCd,
            "format": "json",
            # 부산교통공사 코드
            "railOprIsttCd": "BS",
        }

        endpoint = "https://openapi.kric.go.kr/openapi/trainUseInfo/subwayTimetable"
        res = requests.get(endpoint, params=params)
        schedule = res.json()['body']
        schedule.sort(key=lambda x: x['arvTm'])

        # 4시30분을 기준으로 첫차부터 정렬하는 로직
        schedule.sort(key=lambda x: int(x['arvTm']) if int(
            x['arvTm']) >= 43000 else int(x['arvTm']) + 240000)

        uplines = [s for s in schedule if int(s['trnNo'][0]) % 2 != 0]
        downlines = [s for s in schedule if int(s['trnNo'][0]) % 2 == 0]

        def in_timerange(target: str):
            # 뒤로 15분, 앞으로 1시간 이내의 스케쥴만 추출합니다.
            today = datetime.datetime.today().date()
            target_hour = target[:2]
            target_min = target[2:4]
            target_sec = target[4:]
            # 추출한 시간 정보를 오늘 날짜와 결합하여 datetime 객체를 생성합니다.
            target_datetime = datetime.datetime.strptime(
                f'{today} {target_hour}:{target_min}:{target_sec}', '%Y-%m-%d %H:%M:%S')
            if target_datetime.hour < 4:
                target_datetime = target_datetime + datetime.timedelta(days=1)

            if target_datetime < end_time and target_datetime > start_time:
                print(target_datetime)
                return True
            return False

        target_uplines = [
            uplines[0]] + [s for s in uplines if in_timerange(s['arvTm'])] + uplines[-2:]
        target_downlines = [
            downlines[0]] + [s for s in downlines if in_timerange(s['arvTm'])] + downlines[-2:]

        def extract_time(schedule: dict):
            result = f"{schedule['arvTm'][:2]}시 {schedule['arvTm'][2:4]}분"
            return result

        result = f"{station_name}역의 열차 시간표입니다. 현재시각:{now.strftime('%H:%M:%S')}\n"
        result += "상행선(다대포해수욕장)\n"
        result += "첫차: " + extract_time(target_uplines[0]) + " " + "막차: " + extract_time(
            target_uplines[-2]) + ', ' + extract_time(target_uplines[-1]) + "\n"
        result += "상행선 최근 열차\n" + \
            "\n".join(extract_time(s) for s in target_uplines[1:-2])
        result += "\n하행선(노포)\n"
        result += "첫차: " + extract_time(target_downlines[0]) + " " + "막차: " + extract_time(
            target_downlines[-2]) + ', ' + extract_time(target_downlines[-1]) + "\n"
        result += "하선 최근 열차\n" + \
            "\n".join(extract_time(s) for s in target_downlines[1:-2])

        return result

    return run(station_name)
