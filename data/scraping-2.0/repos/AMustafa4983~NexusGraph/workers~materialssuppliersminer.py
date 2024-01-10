from openai import OpenAI
import os 
class MaterialsSuppliersMiner:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print('MaterialsSuppliersMiner Initialized')

    
    def predict(self, section_text):
        print('MaterialsSuppliersMiner Mining for Materials and Suppliers...')
        prediction = self.client.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature=0.2,
        messages=[
            {'role':'system','content':'''you're an ner system used to extract features from a given text and form it in a JSON file
                                        the output should be in this structure.
                                        you should extract materials and suppliers and experiments and catalog them in this structure.

                                        {
                                        experiments: [
                                        {
                                        'experiment_title1' : 'output_title',
                                        'experiment_items' : [ {'material' : 'output_material' , 'supplier' : 'output_supplier', 'material_usage':'output_material_usage'}, ... ],
                                        },
                                        {
                                        'experiment_title2' : 'output_title',
                                        'experiment_items' : [ {'material' : 'output_material' , 'supplier' : 'output_supplier','material_usage':'output_material_usage'}, ... ],
                                        },
                                        ...
                                        ]
                                        }'''},
            {'role':'user', 'content':section_text[:4050]},
            {'role':'user', 'content':section_text[4050:]},

        ]
    )    

        return prediction