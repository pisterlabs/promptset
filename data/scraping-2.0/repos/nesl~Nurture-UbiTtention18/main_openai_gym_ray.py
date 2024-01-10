from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from constant import *
from environment import *
from behavior import *
from openai_gym.engagement_gym import EngagementGym
from openai_gym.engagement_gym_coach import EngagementGymCoach


if __name__ == "__main__":

    ### simulation configuration
    rewardCriteria = {
            ANSWER_NOTIFICATION_ACCEPT: 1,
            ANSWER_NOTIFICATION_IGNORE: 0,
            ANSWER_NOTIFICATION_DISMISS: -5,
    }
    verbose = False

    ### user habit model

    #environment = AlwaysSayOKUser()
    #environment = StubbornUser()
    #environment = LessStubbornUser()
    #environment = SurveyUser('survey/ver1_pilot/data/02.txt')
    environment = MTurkSurveyUser(filePaths=[
            'survey/ver2_mturk/results/01_1st_Batch_3137574_batch_results.csv',
            'survey/ver2_mturk/results/02_Batch_3148398_batch_results.csv',
            'survey/ver2_mturk/results/03_Batch_3149214_batch_results.csv',
    ], filterFunc=(lambda r: ord(r['rawWorkerID'][-1]) % 3 == 2))

    ### user daily routing modevior = RandomBehavior()

    #behavior = ExtraSensoryBehavior('behavior/data/2.txt')
    #behavior = ExtraSensoryBehavior('behavior/data/4.txt')
    #behavior = ExtraSensoryBehavior('behavior/data/5.txt')
    behavior = ExtraSensoryBehavior('behavior/data/6.txt')

    episodeLengthDay = 7
    stepSizeMinute = 10

    ### here we go (create environment and run)
    env_creator_name = "user-engagement"
    register_env(env_creator_name, lambda config: EngagementGymCoach(config))
    ray.init()
    run_experiments({
        "demo": {
            "run": "PPO",
            "env": "user-engagement",
            "config": {
                "env_config": {
                    "rewardCriteria": rewardCriteria,
                    "environment": environment,
                    "behavior": behavior,
                    "verbose": verbose,
                    "episodeLengthDay": episodeLengthDay,
                    "stepSizeMinute": stepSizeMinute,
                },
            },
        },
    })


