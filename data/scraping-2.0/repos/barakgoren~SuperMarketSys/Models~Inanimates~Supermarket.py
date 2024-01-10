from typing import List
from Models.Entities.Employee import Employee
from Models.Entities.Cashier import Cashier
from Models.Entities.Sadran import Sadran
from Models.Entities.ShiftManager import ShiftManager
from Models.Entities.SuperManager import SuperManager
from Models.Entities.Client import Client
from Models.Inanimates.Shelf import Shelf
from Models.Inanimates.Purchase import Purchase
from Models.Inanimates.PurchaseDisplay import PurchaseDisplay
from Models.Tools.FileHandler import FileHandler
from Models.Tools.Tools import Tools
import sys
import threading
import openai


class Supermarket:

    def __init__(self):
        self._shelves: List[Shelf] = FileHandler.read_shelves_from_directory("Files/Shelves")
        self._workers: List[Employee] = FileHandler.read_employees_from_directory("Files/Employees")
        self._workers.extend(FileHandler.read_shift_managers_from_directory("Files/Employees/ShiftManagers"))
        self._purchases: List[PurchaseDisplay] = FileHandler.read_purchases()

    @staticmethod
    def say_menu(text):
        threading.Thread(target=FileHandler.say, args=(text,)).start()

    @staticmethod
    def talk_to_gpt(messages):
        openai.api_key = Tools.api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = response.choices[0].message.content
        messages.append(response.choices[0].message)
        return message

    def add_purchase(self, purchase: PurchaseDisplay):
        self._purchases.append(purchase)

    def client_menu(self, client: Client):
        print("-------------------------------")
        print(f"Hello {Tools.paint_client(f'{client.get_name()}')}")
        self.say_menu("Choose action")
        user_in = input("-------------------------------\n"
                      "Choose action:"
                      "\n1. Purchase wish list."
                      "\n2. Add item to wish list."
                      "\n3. Show my wish list.\n"
                      "-------------------------------\n-> ")
        try:
            input_number = int(user_in)
        except ValueError:
            print("Thank you for buying in Rami Levi")
            self.say_menu("Thank you for buying in Rami Lavi")
            sys.exit()

        try:

            if input_number == 1:
                print("-------------------------------")
                self.say_menu("Choose cashierr")
                print(Tools.shiftmanager_background("             Choose cashier: "))
                cashiers_list: List[Cashier] = []
                for worker in self._workers:
                    if isinstance(worker, Cashier):
                        cashiers_list.append(worker)

                for i in range(0, len(cashiers_list)):
                    print(Tools.cashier_background(f"{i+1}. {cashiers_list[i].get_name()}"))

                print("-------------------------------")
                cashier_choose = int(input())-1
                if cashier_choose == 5:
                    messages = []
                    list = []
                    for product in self._shelves[0].get_product():
                      list.append(product.get_name())
                    text = "Hi gpt, now i need you to act like a cashier, you are now in a console app of Supermarket, your role" \
                           "is to act like a cashier in front of a client from now on, i give you a list of products " \
                           "in case he asks you to show him what products we have.".join(list)
                    messages.append({"role": "user", "content": text})
                    while True:
                        user_input = input("Talk to AI Agent: ")
                        messages.append({"role": "user", "content": user_input})
                        response = self.talk_to_gpt(messages)
                        print(response)
                        if user_input == 'exit':
                            break

                total_price: float = 0
                for prod in client.get_wishlist():
                    price = float(prod.get_price())
                    total_price = total_price + price
                formatted_number = "{:.2f}".format(total_price)
                self.say_menu(f"Your total is ${formatted_number}, are you sure you want to proceed?")
                ask = input(f"Your total is ${formatted_number}, are you sure you want to proceed? y/n:")
                if ask == 'y':
                    if(total_price == 0):
                        FileHandler.say("You dont have anything to pay for")
                    else:
                        new_purchase = Purchase(client, cashiers_list[cashier_choose])
                        cashiers_list[cashier_choose].purchase_client_products(client)
                        print("Purchase succeed")
                        FileHandler.say("Purchase succeed")
                        FileHandler.add_purchase(new_purchase)
                        self.add_purchase(new_purchase)
                else:

                    print("Ok back to Main Menu!")
                    FileHandler.say("Ok back to Main Menu!")
                    raise ValueError

            if input_number == 2:
                print("-------------------------------")
                self.say_menu("Which shelf would you like to search in?")
                print("Choose Shelf: ")
                for i in range(0, len(self._shelves)):
                    print(f"{i+1}. {self._shelves[i].get_name()}")

                print("-------------------------------")
                shelf_choose = int(input("-> "))-1
                print("-------------------------------")
                self.say_menu("Please choose product")
                print(Tools.shiftmanager_background("             Choose Product: "))
                for i in range(0, len(self._shelves[shelf_choose].get_product())):
                    digits_of_i = len(str(i+1))
                    print(Tools.red_background(str(i+1) + ". ") +
                          Tools.print_products(self._shelves[shelf_choose].get_product()[i], digits_of_i))
                print("-------------------------------")
                product_choose = int(input("-> "))-1
                FileHandler.say(f"You added "
                                f"{self._shelves[shelf_choose].get_product()[product_choose].get_name()} to your cart")

                if self._shelves[shelf_choose].get_product()[product_choose] not in client.get_wishlist():
                    FileHandler.add_to_wishlist_file(
                        f"../PythonFinal/Files/Clients/{client.get_id()}/wanted_products.txt", client,
                        self._shelves[shelf_choose].get_product()[product_choose])
                    client.add_product(self._shelves[shelf_choose].get_product()[product_choose])

            if input_number == 3:
                print("-------------------------------")
                print(Tools.client_purchase("               My cart: "))
                if len(client.get_wishlist()) == 0:
                    print("No items.")
                    FileHandler.say("You have no items in your cart")
                else:
                    FileHandler.say("your list contains")
                    sum_price = 0
                    for product in client.get_wishlist():
                        print(Tools.print_products(product, -2))
                        sum_price += product.get_price()
                        FileHandler.say(product.get_name())
                print(Tools.cashier_background(" "))
                formatted_sum = "{:.2f}".format(sum_price)
                print(Tools.cashier_background(f"Total price:   ${formatted_sum}"))
        except ValueError:
            print("Back to Main Menu")

    def sadran_menu(self, sadran: Sadran):
        print("-------------------------------")
        print(f"Hello {Tools.paint_sadran(f'{sadran.get_name()}')}")
        user_in = input("-------------------------------\n"
                        "Choose action:"
                        "\n1. Add product to shelf."
                        "\n2. Remove product from shelf\n"
                        "-------------------------------\n> ")
        try:
            input_number = int(user_in)
        except ValueError:
            print("Good Bye!")
            self.say_menu("Good bye")
            sys.exit()

        if input_number == 1:
            sadran.add_product(self._shelves)
        if input_number == 2:
            sadran.remove_product(self._shelves)

    def shift_manager_menu(self, shift_manager: ShiftManager):
        print("-------------------------------")
        print(f"Hello {Tools.paint_shiftmanager(f'{shift_manager.get_name()}')}")
        user_in = input("-------------------------------\n"
                        "Choose action:"
                        "\n1. See who is in the store."
                        "\n2. Add product to shelf."
                        "\n3. Remove product from shelf."
                        "\n4. Add worker."
                        "\n5. Remove worker."
                        "\n6. See all purchases."
                        "\n7. Sell to Client.\n"
                        "-------------------------------\n-> ")
        try:
            input_number = int(user_in)
        except ValueError:
            print("Have a good day!")
            self.say_menu("Have a good day!")
            sys.exit()
        try:
            if input_number == 6:
                shift_manager.see_all_purchases(self._purchases)
            if input_number == 4:
                employee_choose = int(input("-------------------------------\n"
                            "Sadran or Cashier?:"
                            "\n1. Sadran."
                            "\n2. Cashier.\n"
                            "-------------------------------\n-> "))

                if employee_choose == 1:
                    shift_manager.add_sadran()
                if employee_choose == 2:
                    shift_manager.add_cashier()
            if input_number == 5:
                shift_manager.remove_employee(self._workers)
            if input_number == 2:
                shift_manager.add_product(self._shelves)
            if input_number == 3:
                shift_manager.remove_product(self._shelves)
            if input_number == 1:
                self.display_all_entities()
            if input_number == 7:
                clients = FileHandler.read_clients("../PythonFinal/Files/Clients")
                purchase = shift_manager.sell_to_client(clients)
                self.add_purchase(purchase)
        except ValueError:
            return

    def super_manager_menu(self, super_manager: SuperManager):
        print("-------------------------------")
        print(f"Hello {Tools.paint_super_manager(f'{super_manager.get_name()}')}")
        user_in = input("-------------------------------\n"
                        "Choose action:"
                        "\n1. See who is in the store."
                        "\n2. Add worker."
                        "\n3. Remove worker."
                        "\n4. See all purchases.\n"
                        "-------------------------------\n-> ")

        try:
            input_num = int(user_in)
        except BaseException:
            return
        try:
            if input_num == 1:
                self.display_all_entities()
            if input_num == 2:
                super_manager.add_shift_manager(self._workers)
            if input_num == 3:
                super_manager.remove_employee(self._workers)
            if input_num == 4:
                if len(self._purchases) == 0:
                    print("No purchases for today!")
                else:
                    sum_prices = 0
                    for purchase in self._purchases:
                        purchase.display_purchase()
                        sum_prices += purchase.get_total()

                    formatted_sum = "{:.2f}".format(sum_prices)
                    print(f"Total for today: ${formatted_sum}")
        except BaseException as e:
            print(e.args)
            return

    def cashier_menu(self, cashier: Cashier):
        print("-------------------------------")
        print(f"Hello {Tools.paint_cashier(f'{cashier.get_name()}')}")
        print("-------------------------------")
        cashier.sell_to_client()


    def display_all_entities(self):
        clients = FileHandler.read_clients("../PythonFinal/Files/Clients")
        for client in clients:
            print("-----------------Client-----------------")
            client.display_client()
        print()
        for employee in self._workers:
            employee.display_employee()

    def start(self):
        # There is Unhandled exception when picking the user!!
        clients = FileHandler.read_clients("../PythonFinal/Files/Clients")
        self.shift_manager_menu(self._workers[13])

