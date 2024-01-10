import openai 

class AnkiDeckGenerator:
    def GenerateAnkiDeck(app):
        list = [[]]
        list = app.getExcel()
        app.updateFileStatus(1)
        newList = [[]]
        newList.append(list[0])
        newList.append(list[1])
        newList.append(list[2])
        longestList = len(newList[1])
        if len(newList[2]) > longestList:
            longestList = len(newList[2])
        if len(newList[3]) > longestList:
            longestList = len(newList[3])

        openai.api_key = ''
        
        message1 = "Write a short simple japanese sentences with the following words Also add translation." + "\n"
        half = longestList * 0.5
        for id in range(int(half)):
            word = str(newList[1][id])
            if word == "*":
                word = str(newList[2][id])
            message1 += word + " "
        
        messageContainer = [ {"role": "system", "content":  
                       message1} ] 
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messageContainer)
        print(chat.choices[0].message.content)
        return

        #TODO
        with open('readme.txt', 'w', encoding="utf-8") as f:
            f.write("#separator:tab" + "\n" + "#html:true" + "\n" + "#tags column:3" + "\n")
            for i in range(longestList):
                word = str(newList[1][i])
                if word == "*":
                    word = str(newList[2][i])
                message = [ {"role": "system", "content":  
                        "Write a short simple japanese sentence with " + word + " Also add translation."} ] 
                chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message)
                result = chat.choices[0].message.content.split("\n")
                print(result)
                f.write(str(newList[1][i]) + "<br>" +  result[0] + ";" + 
                        str(newList[2][i]) + "<br>" + str(newList[3][i]) + "<br>" + 
                        result[1] + "<br>" + "\n")
                app.updateProgressBar(int(i / longestList * 100))
        app.updateFileStatus(2)