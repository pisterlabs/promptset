import webbrowser
import os
import time
from trustee.report.subtree import get_subtree
from trustee.report.format import TrusteeFormatter
import openai
import traceback


class TrusteeThreshold:
    def __init__(self, trust_report, output_directory) -> None:
       
        self.guide = """
        
            'q' to quit 
            'open' to open trustee explanation to web browser
            'target (class index)' to list all branches to that target class
            'target all' to list all branches to all target classes
            'qi (target class)' quart impurity : till the first ancestor with gini value above 25%
            'aic (target class)' average impurity change : avg all the imp change in the branches then print all nodes with less than the avg
            'cus (target class) (custom threshold)' custom : till custom threshold value inputted by the custom_threshold param
            'full (threshold type) (target class)' The full tree for whatever type qi or aic
            
            """
        self.trust_report = trust_report
        self.output_directory = output_directory
        self.formatter = TrusteeFormatter(trust_report, output_directory)
        
    def run(self):
        print(self.guide)
        
        while True:
            user_input = input()
            command = user_input.split()
            
            if len(command) == 1:
                
                if command[0] in ['quit', 'q']:
                    break
                
                elif command[0] == "open":
                    start_time = time.time()
                    filename = 'file:///' + os.getcwd() + '/' + self.output_directory + '/trust_report.html'
                    try: 
                        print("building json file...")
                        self.formatter.json()
                        print("building html file...")
                        self.formatter.html()
                        print("done.")
                    except Exception as e:
                        print("Error building html file:", str(e))
                        traceback.print_exc()
                    try:
                        print(f"opening {filename} in web browser... \n")
                        webbrowser.open_new_tab(filename)
                        print("done.")
                        print("time === ",(time.time() - start_time))
                    except Exception as e:
                        print("Error opening html file:", str(e))
                        traceback.print_exc()
            
                elif command[0] == "help":
                    print(self.guide)      
            #elif len(command) == 2 and command[0] == 'target':
                #if command[1].isdigit():
                    # find all paths to the target class with int given index
                    # print(self.thresholder.find_paths_to_class(int(command[1])))
                #elif command[1] == 'all':
                    # find all paths to all target classes
                    # self.thresholder.all_target_paths()
                    
            # Thresholding
            elif len(command) == 2 and command[0] == "qi" and command[1].isdigit():
                get_subtree(self.trust_report.max_dt, int(command[1]) ,self.trust_report.class_names, self.trust_report.feature_names).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_Trustee_Threshold", format="pdf")
                get_subtree(self.trust_report.max_dt, int(command[1]) ,self.trust_report.class_names, self.trust_report.feature_names).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_Trustee_Threshold", format="png")
            elif len(command) == 2 and command[0] == "aic" and command[1].isdigit():
                get_subtree(self.trust_report.max_dt, int(command[1]) ,self.trust_report.class_names, self.trust_report.feature_names, threshold="avg imp change").render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_Trustee_Threshold", format="pdf")
                get_subtree(self.trust_report.max_dt, int(command[1]) ,self.trust_report.class_names, self.trust_report.feature_names, threshold="avg imp change").render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_Trustee_Threshold", format="png")
            elif len(command) == 3 and command[0] == "cus" and command[1].isdigit() and command[2].isdigit():
                get_subtree(self.trust_report.max_dt, int(command[1]) ,self.trust_report.class_names, self.trust_report.feature_names, threshold="custom", custom_threshold=int(command[2])).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_Trustee_Threshold", format="pdf")
                get_subtree(self.trust_report.max_dt, int(command[1]) ,self.trust_report.class_names, self.trust_report.feature_names, threshold="custom", custom_threshold=int(command[2])).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_Trustee_Threshold", format="png")
            elif len(command) == 3 and command[0] == "full" and command[2].isdigit():
                if command[1] == "qi":
                    get_subtree(self.trust_report.max_dt, int(command[2]),self.trust_report.class_names, self.trust_report.feature_names,threshold="qi" ,full_tree=True).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_Trustee_Threshold", format="pdf")
                    get_subtree(self.trust_report.max_dt, int(command[2]),self.trust_report.class_names, self.trust_report.feature_names,threshold="qi" ,full_tree=True).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_Trustee_Threshold", format="png")
                elif command[1] == "aic":
                    get_subtree(self.trust_report.max_dt, int(command[2]),self.trust_report.class_names, self.trust_report.feature_names,threshold="aic" ,full_tree=True).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_Trustee_Threshold", format="pdf")
                    get_subtree(self.trust_report.max_dt, int(command[2]),self.trust_report.class_names, self.trust_report.feature_names,threshold="aic" ,full_tree=True).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_Trustee_Threshold", format="png")