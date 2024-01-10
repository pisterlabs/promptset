import openai
import chess
import random

class Chat:
    def __init__(self, role="Talented college educated person"):
        """Construct a new Chat

        Args:
            role (str, optional): Define the profile of the person who the chat pretend to be. Defaults to "Talented college educated person".
        """
        openai.api_key = "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"
        self.messages = [{"role": "system", "content": role}]

    def run_conversation(self) -> str:
        """Send the complete set of mesages through the API of the chat

        Returns:
            str: Returns the answer of the chat
        """
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=self.messages,
        )

        response_message = response["choices"][0]["message"].content

        return response_message
    
class ChessChat:
    def __init__(self):
        self.chat = Chat()
        self.board = chess.Board()

    def make_chat_move(self, random_c: str)  -> str:
        """_summary_

        Args:
            random_c (str): _description_

        Returns:
            str: _description_
        """
        new_message = self.chat.run_conversation()
        print(new_message)

        try: 
            self.board.push_san(new_message)
            self.messages.append({"role": "assistant", "content": new_message})
            
        except:
            self.board.push_san(random_c)
            self.messages.append({"role": "assistant", "content": random_c})
        finally:
            return new_message

    def start_chess_game(self, user_player: str, first_move:str) -> str:
        """Function used to determine which of the pieces will be playing the main user, with the first move.

        Args:
            user_player (str): _description_
            first_move (str): _description_

        Returns:
            str: _description_
        """
        if not self.check_move(first_move):
            return "Illegal move"
        
        self.messages = []
        role_chat = {"role": "system", "content": "You are a very talented chess player"}
        
        user_player = user_player.upper
        if user_player == "BLANCAS" or user_player == "BLANCA" or user_player == "WHITE" or user_player == "WHITES":
            movimientos, random_c = self.get_legal_movements()
            
            new_move = {"role": "user", "content": f"Considering that this is the first move and you are playing the white pieces, Which of the following moves do you choose? {movimientos} Give me an answer that only contains the move e.g. d7d6"}
            self.messages.extend([role_chat, new_move])

        else:
            print(self.board.push_san(first_move))
            movimientos, random_c = self.get_legal_movements()
            
            new_move = {"role": "user", "content": f"Considering that this is the first move and you are playing the black pieces, If i move to {first_move} Which of the following moves do you choose? {movimientos} Give me an answer that only contains the move e.g. d7d6"}
            self.messages.extend([role_chat, new_move])

        return self.make_chat_move(random_c)
            
    def get_next_chess_move(self, next_move:str):
        """while

        Args:
            next_move (str): _description_
            inicial_location (str): _description_
            final_location (str): _description_

        Returns:
            _type_: _description_
        """
        if not self.check_move(next_move):
            return -1
        
        print(self.board.push_san(next_move))
        
        movimientos, random_c = self.get_legal_movements()
        new_move = {"role": "user", "content": f"Considering the past moves, If i move to {next_move} which of this moves do you choose? {movimientos} Give me an answer that only contains the move e.g. d7d6"}
        self.messages.append(new_move)

        return self.make_chat_move(random_c)

    def check_for_restrictions(self, move: str):
        """_summary_

        Args:
            move (str): _description_
            inicial_location (str): _description_
            final_location (str): _description_

        Raises:
            Exception: _description_
        """
        if chess.Move.from_uci(move) not in self.board.legal_moves: 
            raise Exception("Ups! El movimiento parece ser ilegal") 
    
    def check_move(self, move) -> bool:
        """_summary_

        Args:
            move (_type_): _description_

        Returns:
            bool: _description_
        """
        return chess.Move.from_uci(move) in self.board.legal_moves
        
    def get_legal_movements(self)  -> tuple:
        """_summary_

        Returns:
            tuple: _description_
        """
        movimientos = ""
        lista_mov = []
        for move in list(self.board.legal_moves):
            movimientos += f"{move}, "
            lista_mov.append(move)
        return movimientos, random.choice(lista_mov)
    
    def get_posible_moves(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        lista_mov = []
        for move in list(self.board.legal_moves):
            lista_mov.append(str(move))
        return lista_mov
    
    def get_posible_moves_by_place(self, place: str) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        moves = self.get_posible_moves()
        posibles = []
        for move in moves:
            if move[0:2] == place:
                posibles.append(move)
        return posibles
    
    def is_check(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return self.board.is_check()
    
    def is_checkmate(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return self.board.is_checkmate()
    
    def get_board(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        boa = str(self.board).replace("\n", " ")
        boa = boa.split(" ")

        board_list = []
        cont = 0
        for i in range(8):
            board_list.append([])
            for _ in range(8):
                board_list[i].append(boa[cont])
                cont += 1

        return board_list
    
    def print_board(self):
        """_summary_
        """
        print(self.board)
    
class CheckChat:   
    def __init__(self):
        self.chat = Chat()

    def check_if_equal(self, question: str, user_answer: str, correct_answer: str) -> bool:
        """_summary_

        Args:
            question (str): _description_
            user_answer (str): _description_
            correct_answer (str): _description_

        Returns:
            bool: _description_
        """
        new_answ = {"role": "user", "content": f"Para la pregunta {question}: dime si la siguiente respuesta {user_answer} es correcta, sabiendo que la respuesta correcta es {correct_answer}. por favor solo responde si o no"}
        
        self.chat.messages.append(new_answ)
        new_message = self.chat.run_conversation()
        self.chat.messages.append({"role": "assistant", "content": new_message})

        respuesta = False
        if ("Si" in new_message) or ("si" in new_message) or ("Sí" in new_message) or ("Si" in new_message):
            respuesta = True

        return respuesta
