#!/usr/bin/env python3

from OpenAIAnalyst import OpenAIAnalyst
from SituationHandler import SituationHandler

if __name__ == "__main__":
    print("Watchdog is on duty...")
    analyst = OpenAIAnalyst("./data/motions/deck")
    
    situationHandler = SituationHandler()
    situationHandler.takeActionOnAnalysis(analyst.analyze_images())
