from tkinter import *
from tkinter import ttk





class GUI:

    def __init__(self):
        pass

    # This method sets the main application window
    # Input: list of users, and a function to perform upon pressing the button
    def mainWindow(self, choice_list, menu_button_action):

        # Main window properties
        window = Tk()
        window.title("Game Recommender")
        window.geometry("300x100")
        window.resizable(False, False)

        # Bottom frame
        bottom = Frame(window)

        # Dropdown menu
        menu = ttk.Combobox(bottom, width=10, font='Arial', state='readonly')
        menu['values'] = choice_list
        menu.pack()

        def buttonAction():
            menu_button_action(menu.get())

        # Recommend button
        menu_button = Button(bottom, width=12, text="Recommend", command=buttonAction)
        menu_button.pack()

        bottom.pack()

        window.mainloop()

    def resultWindow(self, result):
        # Result window
        window = Tk()
        window.title("Recommendations")
        window.geometry("150x225")
        window.resizable(False, False)

        # Results table
        tree = ttk.Treeview(window)
        for i in range(len(result)):
            tree.insert('', i, text=result[i])
        tree.pack()
        window.mainloop()

