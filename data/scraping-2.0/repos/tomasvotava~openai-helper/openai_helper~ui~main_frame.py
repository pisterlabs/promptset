"""UI Components for OpenAI Helper"""

import os
import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from typing import TYPE_CHECKING, Any, Callable, TypedDict

import openai

from openai_helper.context import ContextProvider

if TYPE_CHECKING:
    from openai_helper.ui.app import App

ResultDict = TypedDict("ResultDict", {"result": Any})
ErrorDict = TypedDict("ErrorDict", {"error": str, "exception": Exception})
PresetDict = TypedDict(
    "PresetDict",
    {"regex_whitelist": str, "regex_blacklist": str, "regex_path_whitelist": str, "regex_path_blacklist": str},
)


configuration_presets: dict[str, PresetDict] = {
    "python": {
        "regex_whitelist": r"\.py$|\.toml$|requirements\.txt$|requirements(\.|-)\.txt$",
        "regex_blacklist": r"__\w+__\.py$",
        "regex_path_blacklist": r"\/__\w+__\/|\.venv\/|venv\/",
        "regex_path_whitelist": r"",
    },
}


class BackgroundTask(threading.Thread):
    """Background task"""

    def __init__(self, master: "MainFrame", target, *args, result_queue: queue.Queue | None = None, **kwargs):
        super().__init__()
        self.master = master
        self.result_queue = result_queue or queue.Queue()
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def __call__(
        self,
        on_success: Callable[["MainFrame", ResultDict], None],
        on_error: Callable[["MainFrame", ErrorDict], None] | None = None,
    ):
        """Call the task and run handlers for either success or error.
        The handlers are always passed the result dictionary (or error dictionary on error)
        and the MainFrame instance.
        """
        self.start()
        self.master.after(100, self._check_result, on_success, on_error)

    def _check_result(
        self,
        on_success: Callable[["MainFrame", ResultDict], None],
        on_error: Callable[["MainFrame", ErrorDict], None] | None = None,
    ):
        """Check if the task has finished and run the appropriate handler"""
        if self.result_queue.empty():
            self.master.after(100, self._check_result, on_success, on_error)
            return
        result = self.result_queue.get()
        if "error" in result:
            if on_error:
                on_error(self.master, result)
            else:
                messagebox.showerror(result["error"], str(result["exception"]))
        else:
            on_success(self.master, result)

    def run(self):
        """Run the task"""
        try:
            result = self.target(*self.args, **self.kwargs)
        except Exception as error:  # pylint: disable=broad-except
            self.result_queue.put({"error": "Unexpected error occurred", "exception": error})
            return
        self.result_queue.put({"result": result})


class ModelProviderBackgroundTask(BackgroundTask):
    """Background task providing OpenAI models"""

    def __init__(self, master: "MainFrame", result_queue: queue.Queue | None = None, api_token: str = ""):
        super().__init__(master=master, target=self.list_models, result_queue=result_queue, api_token=api_token)

    def list_models(self, api_token: str):
        """List OpenAI models"""
        openai.api_key = api_token
        return [
            model["id"]
            for model in openai.Model.list()["data"]
            if str(model["root"]).startswith(("code-", "text-", "gpt-"))
        ]


class CompletionAPIBackgroundTask(BackgroundTask):
    """Background task providing an OpenAI completion"""

    def __init__(
        self,
        master: "MainFrame",
        result_queue: queue.Queue | None = None,
        api_token: str = "",
        prompt: str = "",
        context_provider: ContextProvider | None = None,
        max_tokens: int = 500,
        model: str = "davinci",
    ):
        self.api_token = api_token
        self.prompt = prompt
        self.context_provider = context_provider
        self.max_tokens = max_tokens
        self.model = model
        super().__init__(master=master, target=self.get_completion, result_queue=result_queue)

    def get_completion(self):
        """Get completion based on provided prompt"""
        openai.api_key = self.api_token
        return openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": self.prompt},
                {"role": "user", "content": self.context_provider.get_context()},
            ],
            max_tokens=self.max_tokens,
        )


class ModelProviderThread(threading.Thread):
    """Provide OpenAI models"""

    def __init__(self, result_queue: queue.Queue, api_token: str):
        super().__init__()
        self.result_queue = result_queue
        self.api_token = api_token

    def run(self):
        """Provide OpenAI models"""
        openai.api_key = self.api_token
        try:
            models = [
                model["id"]
                for model in openai.Model.list()["data"]
                if str(model["root"]).startswith(("code-", "text-", "gpt-"))
            ]
            self.result_queue.put({"result": sorted(models)})
        except Exception as error:  # pylint: disable=broad-except
            self.result_queue.put({"error": error})


class FileProviderThread(threading.Thread):
    """Thread calculating file paths and their token length"""

    def __init__(
        self,
        result_queue: queue.Queue,
        context_provider: ContextProvider,
    ):
        super().__init__()
        self.result_queue = result_queue
        self.provider = context_provider

    def run(self):
        """Calculate file paths and their token length"""
        try:
            result = [{"tokens": tokens, "path": path} for tokens, path, _ in self.provider.iter_files()]
        except Exception as error:  # pylint: disable=broad-except
            self.result_queue.put({"error": "Unexpected error occurred", "exception": error})
            return
        total_tokens = sum(file["tokens"] for file in result)
        self.result_queue.put({"result": result, "total_tokens": total_tokens})


class CompletionDialog(tk.Toplevel):
    """Dialog for OpenAI completion"""

    def __init__(self, master: "MainFrame", completion: str):
        self.completion = completion
        super().__init__(master)
        self.title("Completion")
        self.textarea = scrolledtext.ScrolledText(self, width=80, height=20)
        self.textarea.insert(tk.END, self.completion)
        self.textarea.pack(fill="both", expand=True)
        self.grab_set()
        self.transient(master.root)


class ProgressDialog(tk.Toplevel):
    """Display a dialog window to make the user wait"""

    def __init__(self, master: "MainFrame", title: str = "Please wait..."):
        super().__init__(master)
        self.title(title)
        ttk.Label(self, text="Please wait, there's an operation under way...").pack()
        self.transient(master.root)


class OpenAISettingsFrame(ttk.Labelframe):
    """OpenAI settings frame"""

    def __init__(self, master: "MainFrame", text: str):
        super().__init__(master, text=text)
        self.root = master
        self.openai_api_token = tk.StringVar(value=self.root.root.configuration.openai_token or "")
        self.selected_model = tk.StringVar(value=self.root.root.configuration.openai_model or "")
        self.max_tokens = tk.StringVar(value=self.root.root.configuration.openai_max_tokens or "500")
        ttk.Label(self, text="OpenAI API token:").grid(column=0, row=0, sticky="w", padx=5)
        ttk.Entry(self, textvariable=self.openai_api_token).grid(column=1, columnspan=5, row=0, sticky=("nsew"), padx=5)

        ttk.Label(self, text="Model:").grid(column=0, row=1, sticky="w", padx=5)
        self.models_combobox = ttk.Combobox(self, textvariable=self.selected_model, values=(), state="readonly")
        self.models_combobox.grid(column=1, row=1, columnspan=4, sticky=("nsew"), padx=5)
        ttk.Button(self, text="Refresh models", command=self.refresh_models).grid(
            column=5, row=1, sticky="ew", padx=5, pady=5
        )

        ttk.Label(self, text="Max tokens:").grid(column=0, row=2, sticky="w", padx=5)
        ttk.Entry(self, textvariable=self.max_tokens).grid(column=1, row=2, sticky=("nsew"), padx=5)

        self.columnconfigure(1, weight=1)
        self.openai_api_token.trace_add(
            "write",
            lambda *_: self.root.update_config("openai_token", self.openai_api_token.get()),
        )

        self.selected_model.trace_add(
            "write",
            lambda *_: self.root.update_config("openai_model", self.selected_model.get()),
        )

        self.max_tokens.trace_add("write", self._trace_max_tokens)

    def _trace_max_tokens(self, *_):
        """Trace max tokens"""
        self.max_tokens.set(str(self.root.extract_int(self.max_tokens.get())))
        self.root.update_config("openai_max_tokens", self.max_tokens.get())

    def refresh_models(self):
        """Refresh list of models"""
        if not self.openai_api_token.get():
            messagebox.showerror("Error", "OpenAI API token is not set")
            return
        self.root.disable()
        task = ModelProviderBackgroundTask(master=self, api_token=self.openai_api_token.get())
        task(lambda _, result: self.update_models_list(result["result"]), self.master.show_background_task_error)

    def update_models_list(self, models: list[str]):
        """Update models list"""
        self.selected_model.set("")
        self.models_combobox.delete(0, "end")
        self.models_combobox["values"] = tuple(sorted(models))
        self.root.enable()


class OptionsFrame(ttk.Labelframe):
    """Options frame"""

    def __init__(self, master: ttk.Widget, text: str):
        super().__init__(master, text=text)
        self.total_tokens = tk.IntVar(value=0)
        self.regex_whitelist = tk.StringVar(value=r"")
        self.regex_blacklist = tk.StringVar(value=r"")
        self.regex_path_whitelist = tk.StringVar(value=r"")
        self.regex_path_blacklist = tk.StringVar(value=r"")
        ttk.Label(self, text="Total tokens:").grid(column=0, row=0, sticky="w", padx=5)
        ttk.Label(self, textvariable=self.total_tokens).grid(column=1, row=0, sticky="w", padx=5)

        ttk.Label(self, text="Regex whitelist:").grid(column=0, row=1, sticky="w", padx=5)
        ttk.Entry(self, textvariable=self.regex_whitelist).grid(column=1, row=1, columnspan=5, sticky=("ew"), padx=5)
        ttk.Label(self, text="Regex blacklist:").grid(column=0, row=2, sticky="w", padx=5)
        ttk.Entry(self, textvariable=self.regex_blacklist).grid(column=1, row=2, columnspan=5, sticky=("ew"), padx=5)
        ttk.Label(self, text="Regex path whitelist:").grid(column=0, row=3, sticky="w", padx=5)
        ttk.Entry(self, textvariable=self.regex_path_whitelist).grid(
            column=1, row=3, columnspan=5, sticky=("ew"), padx=5
        )
        ttk.Label(self, text="Regex path blacklist:").grid(column=0, row=4, sticky="w", padx=5)
        ttk.Entry(self, textvariable=self.regex_path_blacklist).grid(
            column=1, row=4, columnspan=5, sticky=("ew"), padx=5
        )


class FileOptionsFrame(ttk.Labelframe):
    """File options frame"""

    def __init__(self, master: ttk.Widget, text: str):
        super().__init__(master, text=text)
        self.recursive = tk.BooleanVar(value=True)
        self.allow_hidden_subdirectories = tk.BooleanVar(value=False)

        self.skip_unreadable = tk.BooleanVar(value=True)
        self.skip_empty_files = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Recursive", variable=self.recursive).grid(column=0, row=0, sticky="w", padx=5)
        ttk.Checkbutton(self, text="Allow hidden subdirectories", variable=self.allow_hidden_subdirectories).grid(
            column=1, row=0, sticky="w", padx=5
        )
        ttk.Checkbutton(self, text="Skip unreadable files", variable=self.skip_unreadable).grid(
            column=0, row=1, sticky="w", padx=5
        )
        ttk.Checkbutton(self, text="Skip empty files", variable=self.skip_empty_files).grid(
            column=1, row=1, sticky="w", padx=5
        )

        self.grid(column=0, row=5, columnspan=2, sticky="ew", padx=5, pady=5)


class MainFrame(ttk.Frame):
    """Main application frame"""

    def __init__(self, root: "App"):
        super().__init__(root)
        self.root = root
        self._progress: ProgressDialog | None = None

        self.project_path = tk.StringVar(value=os.getcwd())
        self.prompt = tk.StringVar(
            value=(
                "Write a brief README.md file for this project. "
                "I will provide all of the files' contents along with the files' relative paths. "
                "Do not, unless neccessary, comment on individual files but rather on the project's "
                "usage and purpose."
            )
        )

        self.theme = tk.StringVar(value=self.root.configuration.theme or "default")
        self._set_theme()

        self.main_menu = tk.Menu(self.master)
        self.theme_menu = tk.Menu(self.main_menu, tearoff=False)
        for theme in sorted(self.root.themes):
            self.theme_menu.add_radiobutton(label=theme, command=self._set_theme, value=theme, variable=self.theme)
        self.main_menu.add_cascade(label="Theme", menu=self.theme_menu)
        self.root.config(menu=self.main_menu)

        self.preset_menu = tk.Menu(self.main_menu, tearoff=False)
        self._create_preset_menu()
        self.main_menu.add_cascade(label="Presets", menu=self.preset_menu)

        self.pack(fill="both", expand=True)  # , padx=10, pady=10)

        label = ttk.Label(self, text="Project path:")
        path_input = ttk.Entry(self, textvariable=self.project_path)
        button = ttk.Button(self, text="Browse")
        scan_button = ttk.Button(self, text="Scan")

        label.grid(column=0, row=0, sticky="w", padx=5)
        path_input.grid(
            column=1,
            row=0,
            sticky=("ew"),
            padx=5,
        )
        button.grid(column=2, row=0, sticky="e", padx=5)
        scan_button.grid(column=3, row=0, sticky="w", pady=5)

        self.options_frame = OptionsFrame(self, text="Options")
        self.file_options_frame = FileOptionsFrame(self.options_frame, text="File options")

        self.openai_options_frame = OpenAISettingsFrame(self, text="OpenAI")

        self.options_frame.grid(column=0, row=2, columnspan=4, rowspan=6, sticky="ew", padx=5, pady=5)
        self.options_frame.columnconfigure(1, weight=1)
        self.openai_options_frame.grid(column=0, row=8, columnspan=4, sticky="ew", padx=5, pady=5)

        button.bind("<Button-1>", lambda _: self.browse())
        scan_button.bind("<Button-1>", lambda _: self.scan())

        self.columnconfigure(1, weight=1)

        self.filelist = ttk.Treeview(self, columns=("path", "tokens"), show="headings")
        self.filelist.column("path", anchor="w", stretch=True)
        self.filelist.column("tokens", anchor="e", width=100, stretch=False)
        self.filelist.heading("path", text="Path", command=lambda: self._treeview_sort_by_column(self.filelist, "path"))
        self.filelist.heading(
            "tokens",
            text="Tokens",
            command=lambda: self._treeview_sort_by_column(self.filelist, "tokens", numeric=True),
        )

        self.filelist_scroll = ttk.Scrollbar(self, orient="vertical", command=self.filelist.yview)
        self.filelist.grid(column=0, row=1, columnspan=4, sticky=("nsew"), pady=5)
        self.filelist_scroll.grid(column=4, row=1, sticky=("ns"), pady=5)
        self.filelist.configure(yscrollcommand=self.filelist_scroll.set)

        ttk.Label(self, text="Prompt:").grid(column=0, row=11)
        ttk.Entry(self, textvariable=self.prompt).grid(
            column=0, row=12, columnspan=5, rowspan=7, sticky=("nsew"), padx=5, pady=5
        )

        ttk.Button(self, text="Generate", command=self.generate).grid(
            column=0, row=19, columnspan=5, sticky=("ew"), padx=5, pady=5
        )

    def disable(self, title: str = "Please wait..."):
        """Disable all widgets by showing a modal window over the main window"""
        self._progress = ProgressDialog(self, title=title)

    def enable(self):
        """Enable all widgets by closing the modal window"""
        if self._progress:
            self._progress.destroy()
            self._progress = None

    def update_config(self, key: str, value: Any):
        """Update configuration value"""
        setattr(self.root.configuration, key, value)

    def _create_preset_menu(self):
        self.preset_menu.delete(0, "end")
        for preset in sorted(configuration_presets):
            self.preset_menu.add_command(
                label=preset,
                command=lambda preset=preset: self.apply_preset(preset),  # type: ignore
            )
        self.preset_menu.add_separator()
        for preset in sorted(self.root.configuration.presets or {}):
            self.preset_menu.add_command(
                label=preset,
                command=lambda preset=preset: self.apply_preset(preset),  # type: ignore
            )
        self.preset_menu.add_command(label="Save current as preset...", command=self.save_preset)

    def extract_int(self, string: str) -> int:
        """Extract integer from string"""
        return int(re.sub(r"\D", "", string) or "0")

    def save_preset(self):
        """Save current configuration as a preset"""
        preset_name = simpledialog.askstring("Save preset", "Enter preset name")
        if not preset_name:
            return
        self.root.configuration.presets = {
            **(self.root.configuration.presets or {}),
            preset_name: {
                "regex_whitelist": self.options_frame.regex_whitelist.get(),
                "regex_blacklist": self.options_frame.regex_blacklist.get(),
                "regex_path_whitelist": self.options_frame.regex_path_whitelist.get(),
                "regex_path_blacklist": self.options_frame.regex_path_blacklist.get(),
            },
        }
        self._create_preset_menu()

    def apply_preset(self, preset: str | PresetDict):
        """Apply predefined preset either by its name or by its dict"""
        custom_presets: dict[str, PresetDict] = self.root.configuration.presets or {}
        if isinstance(preset, str):
            preset = configuration_presets.get(preset) or custom_presets[preset]
        if not preset:
            return
        self.options_frame.regex_whitelist.set(preset["regex_whitelist"])
        self.options_frame.regex_blacklist.set(preset["regex_blacklist"])
        self.options_frame.regex_path_whitelist.set(preset["regex_path_whitelist"])
        self.options_frame.regex_path_blacklist.set(preset["regex_path_blacklist"])

    def _treeview_sort_by_column(
        self, treeview: ttk.Treeview, col: str, descending: bool = False, numeric: bool = False
    ):
        """Sort treeview by tokens"""
        data = [(treeview.set(child, col), child) for child in treeview.get_children("")]
        data.sort(key=lambda x: int(x[0]) if numeric else x[0], reverse=descending)
        for index, item in enumerate(data):
            treeview.move(item[1], "", index)
        treeview.heading(col, command=lambda: self._treeview_sort_by_column(treeview, col, not descending, numeric))

    def _set_theme(self):
        """Set theme"""
        theme = self.theme.get()
        self.root.set_theme(theme)
        self.root.configuration.theme = theme

    def generate(self):
        """ "Generate completion"""
        if not self.openai_options_frame.openai_api_token.get():
            messagebox.showerror("Error", "OpenAI API token is not set")
            return
        if not self.openai_options_frame.selected_model.get():
            messagebox.showerror("Error", "Model is not selected")
            return
        if not self.prompt.get():
            messagebox.showerror("Error", "Prompt is empty")
            return
        self.disable()
        task = CompletionAPIBackgroundTask(
            master=self.root,
            api_token=self.openai_options_frame.openai_api_token.get(),
            model=self.openai_options_frame.selected_model.get(),
            prompt=self.prompt.get(),
            max_tokens=int(self.openai_options_frame.max_tokens.get()),
            context_provider=self.context_provider,
        )
        task(
            lambda _, result: CompletionDialog(self, result["result"]["choices"][0]["message"]["content"]),
            self.show_background_task_error,
        )

    def show_completion_result(self, _, result: ResultDict):
        """Show completion result"""
        self.enable()
        CompletionDialog(self, result["result"]["choices"][0]["message"]["content"])

    def show_background_task_error(self, _root: "MainFrame", result: ErrorDict):
        """Show background task error"""
        self.enable()
        messagebox.showerror(result["error"], str(result["exception"]))

    def _configure_children(self, **cnf):
        """Configure children"""
        for child in (*self.winfo_children(), *self.filelist.winfo_children()):
            try:
                child.configure(**cnf)
            except tk.TclError:
                continue

    @property
    def context_provider(self) -> ContextProvider:
        """Get context provider"""
        return ContextProvider(
            directory=Path(self.project_path.get()),
            regex_whitelist=self.options_frame.regex_whitelist.get().strip() or None,
            regex_blacklist=self.options_frame.regex_blacklist.get().strip() or None,
            regex_path_whitelist=self.options_frame.regex_path_whitelist.get().strip() or None,
            regex_path_blacklist=self.options_frame.regex_path_blacklist.get().strip() or None,
            recursive=self.file_options_frame.recursive.get(),
            allow_hidden_subdirectories=self.file_options_frame.allow_hidden_subdirectories.get(),
            skip_unreadable=self.file_options_frame.skip_unreadable.get(),
            skip_empty=self.file_options_frame.skip_empty_files.get(),
        )

    def scan(self):
        """Scan for files"""
        self.disable()
        self.filelist.delete(*self.filelist.get_children())
        result_queue = queue.Queue()
        thread = FileProviderThread(
            result_queue=result_queue,
            context_provider=self.context_provider,
        )
        thread.start()
        self.after(100, self._check_queue, result_queue)

    def _check_queue(self, result_queue: queue.Queue):
        """Check if the thread has finished and update the file list"""
        if result_queue.empty():
            self.after(100, self._check_queue, result_queue)
        else:
            result = result_queue.get()
            if "error" in result:
                self.enable()
                messagebox.showerror(result["error"], str(result["exception"]))
                return
            self.options_frame.total_tokens.set(result["total_tokens"])
            for file in result["result"]:
                self.filelist.insert("", "end", values=(file["path"], file["tokens"]))
            self.enable()

    def browse(self):
        """Browse for project path"""
        if path := filedialog.askdirectory(
            initialdir=self.project_path.get(), title="Select project directory", mustexist=True
        ):
            self.project_path.set(path)
