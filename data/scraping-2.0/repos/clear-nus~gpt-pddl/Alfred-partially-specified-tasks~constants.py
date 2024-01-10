import openai

oneshot_problem_goal_pddl_dict = {
    "scene9": """
 
        (:goal (and
                (objectAtLocation Mug1 Plate2)
            )
        )            
""",
    "scene10": """
 
        (:goal (and
                (objectAtLocation CD1 Box1)
            )
        )            
""",
    "scene12": """
 
        (:goal (and
                (objectAtLocation CD1 Box1)
            )
        )            
""",
    "scene11": """
 
        (:goal (and
                (objectAtLocation Pencil1 Box2)
            )
        )            
""",
    "scene13": """
 
        (:goal (and
                (objectAtLocation Laptop1 Sofa2)
            )
        )            
""",
    "scene14": """
 
        (:goal (and
                (objectAtLocation Pencil2 Sofa1)
            )
        )            
""",
    "scene21": """

        (:goal (and
                (objectAtLocation Book3 Sofa1)
            )
        )            
""",
    "scene22": """

        (:goal (and
                (objectAtLocation Book3 Sofa1)
            )
        )            
""",
}

one_shot_nl_dict = {
    "scene9": "Put Mug1 next to Egg1. Do not move Egg1.",
    "scene10": "Move CD1 to the box with two books.",
    "scene12": "Move CD1 to the box with more books.",
    "scene11": "Move Pencil1 to the box with three books.",
    "scene13": "Put Laptop1 on the sofa with Book1. Do not put it in box.",
    "scene14": "Put Pencil2 on the sofa with KeyChain1. Do not put it in box.",
    "scene21": "Put Book3 on a couch.",
    "scene22": "Put Book3 on a couch.",
}

oneshot_problem_nogoal_pddl_dict = {
    "scene9": """
(:objects 
    (Pot1 - object)
    (Spatula1 - object)
    (Cup1 - object)
    (Plate1 - object)
    (Bread1 - object)
    (Egg1 - object)
    (SoapBottle1 - object)
    (Mug1 - object)
    (Tomato1 - object)
    (Potato1 - object)
    (Plate2 - receptacle)
    (CounterTop3 - receptacle)
    (GarbageCan2 - receptacle)
    (Bowl1 - receptacle)
    (CounterTop2 - receptacle)
    (Cabinet1 - receptacle)
    (Plate1 - receptacle)
    (Fridge1 - receptacle)
    (GarbageCan1 - receptacle)
    (CounterTop1 - receptacle)
    (Sink1 - receptacle)
    (Oven1 - receptacle)
    (Microwave1 - receptacle)
    (Floor1 - receptacle)
 )
(:init 
    (sliceable Egg1)
    (sliceable Bread1)
    (sliceable Tomato1)
    (sliceable Potato1)
    (heatable Plate2)
    (heatable Cup1)
    (heatable Plate1)
    (heatable Bread1)
    (heatable Egg1)
    (heatable Mug1)
    (heatable Tomato1)
    (heatable Potato1)
    (coolable Plate2)
    (coolable Bowl1)
    (coolable Pot1)
    (coolable Cup1)
    (coolable Plate1)
    (coolable Bread1)
    (coolable Egg1)
    (coolable Mug1)
    (coolable Tomato1)
    (coolable Potato1)
    (objectAtLocation Cup1 Floor1)
    (objectAtLocation Pot1 Fridge1)
    (objectAtLocation Bread1 CounterTop2)
    (objectAtLocation Mug1 Plate1)
    (objectAtLocation Potato1 CounterTop1)
    (objectAtLocation Tomato1 CounterTop3)
    (objectAtLocation Plate1 Cabinet1)
    (objectAtLocation Egg1 Plate2)
    (objectAtLocation Spatula1 Bowl1)
    (objectAtLocation SoapBottle1 CounterTop1)
 )
""",
    "scene10": """
(:objects 
 	(CreditCard1 - object)
 	(CellPhone2 - object)
 	(CD2 - object)
 	(Book1 - object)
 	(Book3 - object)
 	(CD1 - object)
 	(Pencil1 - object)
 	(CellPhone1 - object)
 	(Box1 - object)
 	(Watch2 - object)
 	(CreditCard2 - object)
 	(Laptop1 - object)
 	(Book2 - object)
 	(Watch1 - object)
 	(GarbageCan1 - receptacle)
 	(Drawer1 - receptacle)
 	(Box2 - receptacle)
 	(Box1 - receptacle)
 	(Sofa2 - receptacle)
 	(Sofa1 - receptacle)
 	(Desk1 - receptacle)
 	(Desk2 - receptacle)
 	(Sofa4 - receptacle)
 	(Box3 - receptacle)
 	(Desk3 - receptacle)
 	(Bed1 - receptacle)
 	(Sofa3 - receptacle)
 	(Box4 - receptacle)
 	(Laptop2 - object)
 )
(:init 
 	(objectAtLocation CreditCard1 Desk1)
 	(objectAtLocation CD1 GarbageCan1)
 	(objectAtLocation Watch1 Desk2)
 	(objectAtLocation CellPhone2 Sofa2)
 	(objectAtLocation CreditCard2 Sofa3)
 	(objectAtLocation Laptop1 Bed1)
 	(objectAtLocation CD2 Desk3)
 	(objectAtLocation Laptop2 Desk1)
 	(objectAtLocation Book3 Box1)
 	(objectAtLocation Book1 Box1)
 	(objectAtLocation Book2 Box2)
 )
""",
    "scene12": """
(:objects 
 	(CreditCard1 - object)
 	(CellPhone2 - object)
 	(CD2 - object)
 	(Book1 - object)
 	(Book3 - object)
 	(CD1 - object)
 	(Pencil1 - object)
 	(CellPhone1 - object)
 	(Box1 - object)
 	(Watch2 - object)
 	(CreditCard2 - object)
 	(Laptop1 - object)
 	(Book2 - object)
 	(Watch1 - object)
 	(GarbageCan1 - receptacle)
 	(Drawer1 - receptacle)
 	(Box2 - receptacle)
 	(Box1 - receptacle)
 	(Sofa2 - receptacle)
 	(Sofa1 - receptacle)
 	(Desk1 - receptacle)
 	(Desk2 - receptacle)
 	(Sofa4 - receptacle)
 	(Box3 - receptacle)
 	(Desk3 - receptacle)
 	(Bed1 - receptacle)
 	(Sofa3 - receptacle)
 	(Box4 - receptacle)
 	(Laptop2 - object)
 )
(:init 
 	(objectAtLocation CreditCard1 Desk1)
 	(objectAtLocation CD1 GarbageCan1)
 	(objectAtLocation Watch1 Desk2)
 	(objectAtLocation CellPhone2 Sofa2)
 	(objectAtLocation CreditCard2 Sofa3)
 	(objectAtLocation Laptop1 Bed1)
 	(objectAtLocation CD2 Desk3)
 	(objectAtLocation Laptop2 Desk1)
 	(objectAtLocation Book3 Box1)
 	(objectAtLocation Book1 Box1)
 	(objectAtLocation Book2 Box2)
 )
""",
    "scene11": """
(:objects 
 	(KeyChain2 - object)
 	(CreditCard1 - object)
 	(Book1 - object)
 	(CD1 - object)
 	(Pencil1 - object)
 	(Watch2 - object)
 	(Watch1 - object)
 	(Laptop1 - object)
 	(Laptop2 - object)
 	(CellPhone1 - object)
 	(Pencil2 - object)
 	(KeyChain1 - object)
 	(CellPhone2 - object)
 	(Desk2 - receptacle)
 	(Drawer1 - receptacle)
 	(Desk1 - receptacle)
 	(Box2 - receptacle)
 	(Drawer3 - receptacle)
 	(GarbageCan2 - receptacle)
 	(Box3 - receptacle)
 	(GarbageCan1 - receptacle)
 	(Box1 - receptacle)
 	(Sofa1 - receptacle)
 	(Desk3 - receptacle)
 	(Drawer2 - receptacle)
 	(Bed1 - receptacle)
 	(Book2 - object)
 	(Book3 - object)
 	(Book4 - object)
 	(Book5 - object)
 	(Book6 - object)
 	(Sofa2 - receptacle)
 )
(:init 
 	(objectAtLocation CreditCard1 Drawer1)
 	(objectAtLocation Pencil1 Desk1)
 	(objectAtLocation CellPhone1 Desk2)
 	(objectAtLocation Pencil2 GarbageCan1)
 	(objectAtLocation CellPhone2 Drawer2)
 	(objectAtLocation Laptop1 Sofa1)
 	(objectAtLocation Watch2 Drawer3)
 	(objectAtLocation Laptop2 Bed1)
 	(objectAtLocation CD1 GarbageCan2)
 	(objectAtLocation KeyChain2 Desk3)
 	(objectAtLocation KeyChain1 Desk3)
 	(objectAtLocation Book3 Box3)
 	(objectAtLocation Book2 Box3)
 	(objectAtLocation Book4 Box2)
 	(objectAtLocation Book6 Box2)
 	(objectAtLocation Book1 Box2)
 	(objectAtLocation Book5 Box1)
 )
    """,
    "scene13": """
(:objects 
 	(KeyChain2 - object)
 	(Pencil1 - object)
 	(KeyChain1 - object)
 	(Laptop1 - object)
 	(Laptop2 - object)
 	(Pen1 - object)
 	(Box1 - object)
 	(CellPhone1 - object)
 	(Book1 - object)
 	(Box2 - object)
 	(Book2 - object)
 	(Pen2 - object)
 	(Drawer3 - receptacle)
 	(Sofa5 - receptacle)
 	(Desk2 - receptacle)
 	(Sofa1 - receptacle)
 	(Desk1 - receptacle)
 	(GarbageCan1 - receptacle)
 	(Drawer1 - receptacle)
 	(Sofa2 - receptacle)
 	(Bed1 - receptacle)
 	(Drawer2 - receptacle)
 	(Sofa3 - receptacle)
 	(Sofa4 - receptacle)
 	(Box1 - receptacle_object)
 	(Box2 - receptacle_object)
 	(Box3 - receptacle_object)
 	(Box4 - receptacle_object)
 )
(:init 
 	(objectAtLocation KeyChain1 Drawer1)
 	(objectAtLocation Pencil1 GarbageCan1)
 	(objectAtLocation CellPhone1 Sofa2)
 	(objectAtLocation KeyChain2 Sofa3)
 	(objectAtLocation Pen1 Desk2)
 	(objectAtLocation Pen2 Drawer2)
 	(objectAtLocation Book2 Drawer3)
 	(objectAtLocation Laptop1 Bed1)
 	(objectAtLocation Laptop2 Sofa4)
 	(objectAtLocation Book1 Box2)
 	(objectAtLocation Box1 Sofa1)
 	(objectAtLocation Box2 Sofa2)
 	(objectAtLocation Box3 Sofa3)
 	(objectAtLocation Box4 Sofa3)
 )
""",
    "scene14": """
(:objects 
 	(CreditCard2 - object)
 	(KeyChain2 - object)
 	(Watch1 - object)
 	(Pencil1 - object)
 	(KeyChain1 - object)
 	(Pen1 - object)
 	(CellPhone1 - object)
 	(Book1 - object)
 	(Pillow1 - object)
 	(Book2 - object)
 	(Pencil2 - object)
 	(CreditCard1 - object)
 	(Box2 - receptacle_object)
 	(Box1 - receptacle_object)
 	(Desk2 - receptacle)
 	(Sofa1 - receptacle)
 	(Desk1 - receptacle)
 	(GarbageCan1 - receptacle)
 	(Drawer1 - receptacle)
 	(Sofa2 - receptacle)
 	(Bed1 - receptacle)
 	(Drawer4 - receptacle)
 	(Drawer2 - receptacle)
 	(Drawer3 - receptacle)
 	(AlarmClock1 - object)
 	(Box3 - receptacle_object)
 	(Box4 - receptacle_object)
 	(Sofa3 - receptacle)
 )
(:init 
 	(objectAtLocation Book1 Desk1)
 	(objectAtLocation CreditCard1 Desk2)
 	(objectAtLocation Pillow1 Sofa1)
 	(objectAtLocation Pencil1 GarbageCan1)
 	(objectAtLocation Watch1 Drawer1)
 	(objectAtLocation Pencil2 Drawer2)
 	(objectAtLocation Book2 Drawer3)
 	(objectAtLocation Pen1 Drawer4)
 	(objectAtLocation CellPhone1 Bed1)
 	(objectAtLocation KeyChain1 Sofa2)
 	(objectAtLocation AlarmClock1 Desk2)
 	(objectAtLocation KeyChain1 Box2)
 	(objectAtLocation Box2 Box3)
 	(objectAtLocation Box1 Sofa2)
 	(objectAtLocation Box3 Sofa1)
 	(objectAtLocation Box4 Sofa1)
 )
    """,
    "scene21": """
(:objects 
 	(Pencil1 - object)
 	(CD1 - object)
 	(Laptop1 - object)
 	(KeyChain1 - object)
 	(Pen1 - object)
 	(CellPhone1 - object)
 	(Book1 - object)
 	(Pillow1 - object)
 	(Book2 - object)
 	(Pen2 - object)
 	(Pencil2 - object)
 	(Pen3 - object)
 	(AlarmClock1 - object)
 	(CreditCard1 - object)
 	(Desk4 - receptacle)
 	(Desk3 - receptacle)
 	(GarbageCan2 - receptacle)
 	(Bed2 - receptacle)
 	(Desk2 - receptacle)
 	(Bed3 - receptacle)
 	(Sofa1 - receptacle)
 	(Sofa2 - receptacle)
 	(Desk1 - receptacle)
 	(GarbageCan1 - receptacle)
 	(Desk5 - receptacle)
 	(Drawer1 - receptacle)
 	(Drawer2 - receptacle)
 	(Bed1 - receptacle)
 	(Candle1 - object)
 	(Book3 - object)
 	(Book4 - object)
 	(Book5 - object)
 	(KeyChain2 - object)
 	(Watch1 - object)
 	(Watch2 - object)
 	(Floor1 - receptacle)
 	(Sofa3 - receptacle)
 )
(:init 
 	(objectAtLocation Pen1 GarbageCan1)
 	(objectAtLocation Laptop1 Desk1)
 	(objectAtLocation Pen2 Drawer1)
 	(objectAtLocation CellPhone1 Bed1)
 	(objectAtLocation CD1 GarbageCan2)
 	(objectAtLocation Book1 Desk2)
 	(objectAtLocation Pencil1 Drawer2)
 	(objectAtLocation AlarmClock1 Desk3)
 	(objectAtLocation KeyChain1 Sofa1)
 	(objectAtLocation Pencil2 Desk4)
 	(objectAtLocation Pillow1 Bed2)
 	(objectAtLocation Pen3 Desk5)
 	(objectAtLocation Book2 Bed3)
 	(objectAtLocation CreditCard1 Sofa2)
 	(objectAtLocation Candle1 Desk4)
 	(objectAtLocation Book3 Desk4)
 	(objectAtLocation Book4 Drawer2)
 	(objectAtLocation Book5 Bed1)
 	(objectAtLocation KeyChain2 Desk5)
 	(objectAtLocation Watch1 Desk2)
 	(objectAtLocation Watch2 Desk2)
 )
""",
    "scene22": """
(:objects 
 	(Pencil1 - object)
 	(CD1 - object)
 	(Laptop1 - object)
 	(KeyChain1 - object)
 	(Pen1 - object)
 	(CellPhone1 - object)
 	(Book1 - object)
 	(Pillow1 - object)
 	(Book2 - object)
 	(Pen2 - object)
 	(Pencil2 - object)
 	(Pen3 - object)
 	(AlarmClock1 - object)
 	(CreditCard1 - object)
 	(Desk4 - receptacle)
 	(Desk3 - receptacle)
 	(GarbageCan2 - receptacle)
 	(Bed2 - receptacle)
 	(Desk2 - receptacle)
 	(Bed3 - receptacle)
 	(Sofa1 - receptacle)
 	(Sofa2 - receptacle)
 	(Desk1 - receptacle)
 	(GarbageCan1 - receptacle)
 	(Desk5 - receptacle)
 	(Drawer1 - receptacle)
 	(Drawer2 - receptacle)
 	(Bed1 - receptacle)
 	(Candle1 - object)
 	(Book3 - object)
 	(Book4 - object)
 	(Book5 - object)
 	(KeyChain2 - object)
 	(Watch1 - object)
 	(Watch2 - object)
 	(Floor1 - receptacle)
 	(Sofa3 - receptacle)
 )
(:init 
 	(objectAtLocation Pen1 GarbageCan1)
 	(objectAtLocation Laptop1 Desk1)
 	(objectAtLocation Pen2 Drawer1)
 	(objectAtLocation CellPhone1 Bed1)
 	(objectAtLocation CD1 GarbageCan2)
 	(objectAtLocation Book1 Desk2)
 	(objectAtLocation Pencil1 Drawer2)
 	(objectAtLocation AlarmClock1 Desk3)
 	(objectAtLocation KeyChain1 Sofa1)
 	(objectAtLocation Pencil2 Desk4)
 	(objectAtLocation Pillow1 Bed2)
 	(objectAtLocation Pen3 Desk5)
 	(objectAtLocation Book2 Bed3)
 	(objectAtLocation CreditCard1 Sofa2)
 	(objectAtLocation Candle1 Desk4)
 	(objectAtLocation Book3 Desk4)
 	(objectAtLocation Book4 Drawer2)
 	(objectAtLocation Book5 Bed1)
 	(objectAtLocation KeyChain2 Desk5)
 	(objectAtLocation Watch1 Desk2)
 	(objectAtLocation Watch2 Desk2)
 )
""",
}


synonym_to_type_dict = {
    "timepiece": "Watch",
    "couch": "Sofa",
    "settee": "Sofa",
    "telephone": "CellPhone",
    "mobile phone": "CellPhone",
    "dustbin": "GarbageCan",
    "trash bin": "GarbageCan",
    "keyring": "KeyChain",
}


nl_questions = {
    "scene22": [
        "Put a {} on Desk1.",
        "Is there a {} in the environment?",
        "Put a {} on Desk1. Which object should we put on Desk1?",
    ],
    "scene9": [
        "Put {} next to {}. Do not move {}.",
        "Where is {} in the initial state?",
        "Put {} next to {}. Do not move {}. Where should we put {}?",
    ],
    "scene10": [
        "Move {} to the box with two {}s.",
        "Which box has two {}s in the initial state?",
        "Move {} to the box with two {}s. Which object should we move {} into?",
    ],
    "scene11": [
        "Move {} to the box with three {}s.",
        "Which box has three {}s in the initial state?",
        "Move {} to the box with three {}s. Which object should we move {} into?",
    ],
    "scene12": [
        "Move {} to the box with more {}s.",
        "Which box has more {}s in the initial state?",
        "Move {} to the box with more {}s. Which object should we move {} into?",
    ],
    "scene13": [
        "Put the {} on the sofa with {}. Do not put it in box.",
        "Which sofa is {} in?",
        "Put the {}s on the sofa with {}. Do not put it in box. Which object should we put {} on?",
    ],
    "scene14": [
        "Put the {} on the sofa with {}. Do not put it in box.",
        "Which sofa is {} in?",
        "Put the {} on the sofa with {}. Do not put it in box. Which object should we put {} on?",
    ],
}


openai.api_key = "YOUR_OWN_KEY"
