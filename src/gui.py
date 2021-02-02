# This file contains the GUI class.  The only purpose of this file is to display
# the parameter selection GUI, then provide an object that the main script can
# use to access the selected values from.

from enum import Enum
from tkinter import * 
from tkinter import ttk, filedialog

# This enum contains the four different file paths being set by the GUI.  This 
# approach allows us to write a single function to set the path variable, rather
# than having to create duplicate functions for each path.  We use the values in
# this enum as references as to which we are currently trying to set.
class FileType(Enum):
    ECAD = 0
    HTWO = 1
    ECAD_MODEL = 2
    HTWO_MODEL = 3

# Everything else in this file is the definition of the GUI class
class GUI():
    # This function is the "constructor" which is run when we create the GUI
    # object from the main script.  We can specify default values for the 
    # parameter controls in the GUI.
    def __init__(self, ecad_path="", h2_path="",focus_range=0,mu0=0,background=0,ecad_model_path="",h2_model_path="",save_focus=True,save_norm=True,save_prob=True):
        # Initialising some variables.  Using the "self." notation means these
        # values will be associated with the current instance of the GUI object.
        # We are able to access them from any function within this object using
        # the "self." notation.
        self.ecad_path = ecad_path
        self.h2_path = h2_path
        self.focus_range = focus_range
        self.mu0 = mu0
        self.background = background
        self.ecad_model_path = ecad_model_path
        self.h2_model_path = h2_model_path
        self.save_focus = save_focus
        self.save_norm = save_norm
        self.save_prob = save_prob

    # This is the main function that we call from the main script.  It's the one
    # that will create and display the GUI.
    def run(self):
        # Creating GUI window and specifying some parameters, such as the 
        # window title, dimensions and relative weights of the two columns.  The
        # interface is set up as two columns - the left states the variable name
        # and the right is where the user inputs the relevant values.  The 
        # weights listed below correspond to how wide that column is (although
        # they didn't seem to behave as I expected).
        root = Tk() 
        root.title("Settings") 
        root.geometry("420x600") 
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=4)

        # Adding controls row-by-row.  Since all controls can be classified as 
        # one of three types (file selection, text entry or tickboxes) I created
        # separate functions to generate each of these.  Each function creates
        # one instance of a control with a label on the left and parameter on 
        # the right.  After each control, I increment the row index by one, so 
        # the next control goes on the next line.  These functions run from this
        # specific instance of the GUI, which gives them access to all the 
        # variables set in the constructor.  Such functions are accessed with 
        # the "self." notation.
        row = 0
        self._create_heading(root, row, "File selection")
        row += 1
        self._create_file_selection(root, row, FileType.ECAD, "Select Ecad file", default=self.ecad_path)
        row += 1
        self._create_file_selection(root, row, FileType.HTWO, "Select H2 file", default=self.h2_path)

        row += 1
        self._create_heading(root, row, "Focus parameters")
        row += 1
        self._focus_range_en = self._create_input(root, row, "Focus range", default=self.focus_range)

        row += 1
        self._create_heading(root, row, "Normalisation parameters")
        row += 1
        self._mu0_en = self._create_input(root, row, "mu0", default=self.mu0)
        row += 1
        self._background_en = self._create_input(root, row, "Background", default=self.background)

        row += 1
        self._create_heading(root, row, "WEKA parameters")
        row += 1
        self._create_file_selection(root, row, FileType.ECAD_MODEL, "Select Ecad model file", default=self.ecad_model_path)
        row += 1
        self._create_file_selection(root, row, FileType.HTWO_MODEL, "Select H2 model file", default=self.h2_model_path)

        row += 1
        self._create_heading(root, row, "File output")
        row += 1
        self._save_focus_cb = self._create_checkbutton(root, row, "Save focus images", default=self.save_focus)
        row += 1
        self._save_norm_cb = self._create_checkbutton(root, row, "Save normalised images", default=self.save_norm)
        row += 1
        self._save_prob_cb = self._create_checkbutton(root, row, "Save probabiliy images", default=self.save_prob)

        row += 1
        self._create_process_button(root, row)

        # Starting GUI loop.  This will continue until the "Process" button is 
        # pressed, at which point the GUI will be closed and the main script 
        # will continue processing.
        root.mainloop() 

    # Creates a plain text heading, which spans both columns.  I used these to
    # break the GUI down into sections.
    def _create_heading(self, root, row, text):
        label = Label(root,text=text,font="Calibri 11 bold")
        label.grid(row=row,column=0,columnspan=2,sticky='we', padx=10, pady=5)

    # Creates a file selection control.  On the left is the parameter name and
    # on the right is a button to select the file.
    def _create_file_selection(self, root, row, file_type, text, default=""):
        label = Label(root,text=text)

        # Creating an instance of a Button
        button = ttk.Button(root)

        # Setting the text of this button to be the current file path.  This is
        # done using the _set_button_text function.
        self._set_button_text(button,default)

        # Setting the command that will run when the file selection button is 
        # clicked.  The "lambda" notation tells Python this is a command and
        # allows us to pass argument to the _select_file function.  We actually
        # pass the button as an argument, so the function can update the text
        # on the button to the currently-selected filename.
        button.config(command = lambda: self._select_file(button, file_type))
        
        # Setting the grid position of the label and button.
        label.grid(row=row,column=0,sticky='we', padx=10, pady=5)
        button.grid(row=row,column=1, sticky='we', padx=10, pady=5)

    # Displays a file selection window.  The selected path will be used to 
    # update the text displayed on the button (this is just for usability as it
    # lets the user know what file has been selected).  
    def _select_file(self, button, file_type):
        path = filedialog.askopenfilename()
        self._set_button_text(button, path)

        # Depending on the enum passed as an argument, the relevant path will be
        # updated to the selected location.
        if file_type == FileType.ECAD:
            self.ecad_path = path
        elif file_type == FileType.HTWO:
            self.h2_path = path
        elif file_type == FileType.ECAD_MODEL:
            self.ecad_model_path = path
        elif file_type == FileType.HTWO_MODEL:
            self.h2_model_path = path
    
    # This sets the text of each button.  If it's less than 25 characters long
    # the whole text is displayed.  Otherwise, it's a shortened form.
    def _set_button_text(self, button, text):
        if len(text) < 25:
            button['text'] = text
        else:
            button['text'] = "... %s" % text[-25:]

    # Creates a simple textbox style input, with the parameter name on the left
    # and the parameter value on the right.  The GUI doesn't need to respond to
    # these values being changed, as we'll just access and store them when the
    # user clicks the "Process" button.
    def _create_input(self, root, row, text, default=""):
        label = Label(root,text=text)
        entry = Entry(root)
        entry.insert(0,default)
        
        label.grid(row=row,column=0,sticky='we', padx=10, pady=5)
        entry.grid(row=row,column=1,sticky='we', padx=10, pady=5)

        return entry
    # Creates a tickbox (a.k.a. "checkbutton") control with the paramtere name 
    # on the left and tickbox on the right.  As with the basic text input, we
    # wait until we're collecting values at the end to access the state of this
    # control.
    def _create_checkbutton(self, root, row, text, default=True):
        # The checkbutton control can't take a plain True/False value.  Rather, 
        # we have to convert it to a BooleanVar class object first.
        chk_value = BooleanVar() 
        chk_value.set(default)

        label = Label(root,text=text)
        checkbutton = Checkbutton(root, var=chk_value)

        label.grid(row=row,column=0,sticky='we', padx=10, pady=5)
        checkbutton.grid(row=row,column=1,sticky='we', padx=10, pady=5)

        return chk_value

    # Creates a single button, which when pressed will close the GUI and store
    # the current parameter values.  Unlike the other controls, this spans two
    # columns of the layout grid.
    def _create_process_button(self, root, row):
        button = ttk.Button(root, text="Process", command=lambda: self._process(root))
        button.grid(row=row, column=0, columnspan=2, sticky='we', padx=10, pady=5)

    # When the "Process" button is pressed this function will run.  It closes 
    # the GUI and stores the parameter values within the GUI object (i.e. the 
    # interface is closed, but the object still remains active, so the main
    # script can access it).
    def _process(self, root):
        self.focus_range = self._focus_range_en.get()
        self.mu0 = self._mu0_en.get()
        self.background = self._background_en.get()
        self.save_focus = self._save_focus_cb.get()
        self.save_norm = self._save_norm_cb.get()
        self.save_prob = self._save_prob_cb.get()

        # Using the destroy command to close all GUI windows and allow the 
        # script to proceed.
        root.destroy()
        