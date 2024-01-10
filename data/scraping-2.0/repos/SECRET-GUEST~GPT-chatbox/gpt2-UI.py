#      |               |                                 |

#                  |                                                |            |                                   |                               |                           |                               |                           |                               |                                |                               |
#           |                                  |                                     |

#                  |                     |                                                     |                                   |                               |     |                                   |                               |     |                                   |                               |          |                                   |                               |
#   |                          |                       |                    |
#           |                                    |                                         |     |                    |                                    | |     |                    |                                    | |     |                    |                                    |      |     |                                                        |
#      |                        |                                 |      |                                |                       |                    |                |                       |                    |                |                       |                    |                     |                       |                    |

#                      |                                                |      |                                   |                               |                         |                               |                         |                               |                              |                               |
#           |                                   |                               |                   |                                |                       |                    |          |                                |                       |                    |          |                                |                       |                    |               |                                |                                        |

#                  |                     |
#   |                                |                       |                    |            |                                |                       |                    |     |                                |                       |                    |     |                                |                       |                              |                                |
#           |                               |                                         |                              |                       |                    |                           |                       |                    |                           |                       |                    |                                |                       |                    |

#   |         |                                   PRESENTATION                                                |                                |                    |   |                                |                    |   |                                |                    |        |                                |                    |
#                                                                                                                               |                               |     |                                   |                               |     |                                   |                               |          |                                   |                               |
#                |                            |                 |                          |                                                                              |                               |     |                                   |                               |     |                                   |                               |          |                                   |                               |

#        |                     This little tool use GPT-2 from OpenAI to provide you 
#                                  Answers about general questions you could ask
#                                                VERSION : pre-Alpha


#      |                        |                                         |                                |                       |                    |                |                       |                    |                |                       |                    |                    |                                           |

#                   |            |                   Anyway          |                        |                                         |                                |                       |                    |                |                       |                    |                |                       |                    |                    |

#              |                      |                 have                                                 |                        |                                         |                                |                       |                    |                |                       |                    |                |                       |                    |                     |                                           |


#                 |                                  FUN         |                        |                                         |                                |                       |                    |                |                       |                    |                |                       |                    |                    |                                                                      |

#                                                         =)

#
#                                |                      or                                       |                                                            |                    |                |                       |                    |                |                       |                    |                    |                                           |


#              |                              |       DIE ! !         |                       |                            |                |                       |                    |                |                       |                    |                    |                                           |#      |                        |                                         |                                |

#
#                                                     !                                       |                                |                    |  |                                |                    |  |                                |                    |       |                                |                    |

#      |               |                                 !

#                  |                                                |            |                                   |                               |                           |                               |                           |                               |                                |                               |
#           |                                  !                                     |

#                  |                     |                                                     |                                   |                               |     |                                   |                               |     |                                   |                               |          |                                   |                               |
#   |                          |                       |                    |
#           |                                    |                                         |     |                    |                                    | |     |                    |                                    | |     |                    |                                    |      |     |                    |                                    |
#      |                        |                                 |      |                                |                       |                    |                |                       |                    |                |                       |                    |                    |                                           |

#                      |                                                |      |                                   |                               |                         |                               |                         |                               |                              |                               |
#           |                                   |                               |                   |                                |                       |                    |          |                                |                       |                    |          |                                |                       |                    |               |                                                                |


#____ _    ____ _  _ ____ _  _ ___  ____ _ ____
#|__| |    |___  \/  |__| |\ | |  \ |__/ | |__|
#|  | |___ |___ _/\_ |  | | \| |__/ |  \ | |  |

import os
import sys
import torch
import tkinter as tk
from tkinter import filedialog
from transformers import GPT2Tokenizer, GPT2LMHeadModel


#___  ____ _ _ _ ____ ____    ___  _    ____ _  _ ___
#|__] |  | | | | |___ |__/    |__] |    |__| |\ |  |
#|    |__| |_|_| |___ |  \    |    |___ |  | | \|  |



#OPENING | https://youtu.be/_85LaeTCtV8 :3

#Parametres principaux de notre fenetre de chat
def main():

    global mood, font, main_frame

    master = tk.Tk()
    master.geometry("800x800")
    master.title("Chat with GPT-2")
    master.resizable(width=False, height=False)

#La variable qui va recevoir notre question
    mood = tk.StringVar()

#On parametre la taille de la police de TOUT les widgets (psq flemme)
    font = ("Arial", 18)

#On fait en sorte de match les éléments au sein de notre fenetre tkinter
    main_frame = chat(master)
    main_frame.pack(fill=tk.BOTH, expand=True)

    bottom_frame = submit(master, chat=main_frame.chat)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)


#On lance la loop de création de la fenetre
    master.mainloop()

    
#Fonction utile pour créer notre .exe avec auto-py-to-exe
def ressource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)
#Pour que cela fonctionne, vous devez renommer tous vos paths en ajoutant ressource_path() devant
#Exemple : /icon/lol.png  BECOME  ressource_path(/icon/lol.png)


##Parametre de l'icone de la fenetre tkinter (enlevée pour ...flemme)
#program_directory = sys.path[0]
#master.iconphoto(True, PhotoImage(
#    file=os.path.join(program_directory, ressource_path("ico/sad.png"))))


#Création du token depuis le model préentrainé "GPT-2" 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")



#___  ____ ____ ____ ____ ____ _  _
#|__] |__/ |  | | __ |__/ |__| |\/|
#|    |  \ |__| |__] |  \ |  | |  |


#Fenetre d'envois des questions à GPT-2
class submit(tk.Frame):
    def __init__(self, master=None, chat=None, **kwargs):
        super().__init__(master, **kwargs)
        self.chat = chat
        self.answer_is_ready = False

#Parametres principaux :
        self.config(bg='gray38')

#L'endroit ou l'on écrira notre question :
        self.expected = tk.Entry(

            self,
            textvariable=mood,
            width=48,
            font=font,
            fg='white',
            bg='gray28'
        )

        self.expected.pack(side=tk.LEFT, padx=10, pady=5)

#Charge les images avant leur utilisation (CHANGER LE PATH POUR LE VOTRE en gardant les \\)
        self.icon_send = tk.PhotoImage(
            file='C:\\YOUR\\PATH\\FOR\\THE\\PICTURES\\HERE\\send.png')
        self.icon_save = tk.PhotoImage(
            file='C:\\YOUR\\PATH\\FOR\\THE\\PICTURES\\HERE\\save.png')

#Creation de notre boutton d'envois des réponse
        self.submit = tk.Button(

            self,
            image=self.icon_send,
            command=self.submitting,
            font=font,
            width=50,
            bg='gray28'
        )

        self.submit.pack(side=tk.LEFT, padx=10, pady=5)

#Creation de notre boutton de sauvegarde de la conversation
        self.saver = tk.Button(

            self, image=self.icon_save,
            command=self.saving,
            font=font,
            width=50,
            bg='#EDC51C'
        )

        self.saver.pack(side=tk.LEFT, padx=10, pady=5)


#Fonction de demande de traitement de notre question à GPT-2
    def submitting(self):

        global answer
        question = self.expected.get()

        input_ids = torch.tensor(tokenizer.encode(question)).unsqueeze(0)
        attention_mask = torch.ones(input_ids.shape).to(input_ids.device)

#On paramètre la réponse pour éviter que GPT-2 ne foute le feu au pc
        generated_text = model.generate(

            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_length=250,
            top_k=50,
            top_p=0.95,
            temperature=0.7

        )

#Inclut otre réponse dans une liste
        answer = tokenizer.decode(generated_text.tolist()[0])
        self.answer_is_ready = True 
        self.check_answer()

#Ajoute la réponse au début du chat
        self.chat.insert(tk.END,  "\n\n" + "Ma question : " + answer + "\n\n")

#Faites défiler vers le bas pour voir la dernière réponse
        self.chat.see(tk.END)

    def check_answer(self):

        if self.answer_is_ready:
            self.answer_is_ready = False
            
        self.after(1000, self.check_answer)


#Fonction qui permet de sauvegarder le dernier message ce qui est inutile,
#ca fait partie des améliorations en cours de dev, je pense créer un fichier
#texte tout simplement en le raffraichissant avec les nouvelles réponses
    def saving(self):
        file = filedialog.asksaveasfile(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")]
        )

        if file:
            file.write(answer)
            file.close()




#____ _  _ ____ ___ ___  ____ _  _ 
#|    |__| |__|  |  |__] |  |  \/  
#|___ |  | |  |  |  |__] |__| _/\_ 
                                  

#Fenetre de chat
class chat(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

#Parametre generaux
        self.config(bg='gray10')

#Dans cette frame on a notre retour de la conversation avec le chat
        self.chat_canvas = tk.Frame(
            self,
            bg='gray10',
        )

        self.chat_canvas.pack(padx=2, pady=2)


#Scrollbar (barre de défilement)
        self.scrollbar = tk.Scrollbar(self)
        self.scrollbar.config(width=0, background=self.cget("background"))
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

#Affichage des messages
        self.chat = tk.Text(
            self.chat_canvas, yscrollcommand=self.scrollbar.set)
        self.chat.config(fg='white', bg='gray38', font=font)
        self.chat.pack(fill=tk.BOTH, expand=True)

#Association de la Scrollbar au Texte
        self.scrollbar.config(command=self.chat.see)



#____ ____ ____ _  _ ____ ___    _    ____ _  _ _  _ ____ _  _
#|__/ |  | |    |_/  |___  |     |    |__| |  | |\ | |    |__|
#|  \ |__| |___ | \_ |___  |     |___ |  | |__| | \| |___ |  |

#ENDING | https://www.youtube.com/watch?v=fBkbfSzKIOQ&list=PLA0JNlLE2GW1vzI7vnBrgP6KE3faAHxPR&index=1&ab_channel=SECRETGUEST :3

if __name__ == '__main__':
    main()
