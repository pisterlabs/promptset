import os
import streamlit as st
import streamlit as st
from langchain.callbacks import get_openai_callback
import os
import base64
import pandas as pd



class FileManager:
    def __init__(self, *directories):
        self.directories = directories

    def list_all_files(self):
        all_files = []
        for directory in self.directories:
            for subdir, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(subdir, file)
                    all_files.append(file_path)
        return all_files
    
        
    def preview_file(self, file_path):
        """Preview the content of the selected file based on its type."""
        filename = os.path.basename(file_path)
        file_extension = os.path.splitext(filename)[1].lower()

        # Get file size in MB
        file_size = os.path.getsize(file_path) / (1024 * 1024)

        # Preview for image files
        if file_extension in ['.png', '.jpg', '.jpeg']:
            st.image(file_path)

        # Preview for text files
        elif file_extension in ['.txt', '.md', '.log']:
            with open(file_path, 'r') as f:
                st.text(f.read())

        # Preview for CSV files
        elif file_extension == '.csv':
            if file_size > 50:  # If file size is greater than 50 MB, read only first few rows
                df = pd.read_csv(file_path, nrows=1000)
            else:
                df = pd.read_csv(file_path)
            st.write(df)

        # Add more preview cases for other file types as needed

        else:
            st.warning(f"No preview available for {filename}.")
                
    def delete_file(self, file_path):
        try:
            os.remove(file_path)
            st.success(f"File {file_path} deleted successfully!")
            return True
        except Exception as e:
            st.error(f"Error deleting file: {e}")
            return False
        
    def file_download_button(self, path, label="Download"):
        """Generate a button to download the file."""
        file_path = path
        filename = os.path.basename(file_path)  # extract the filename
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()  # bytes to base64 string
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}" style="display:inline-block;padding:0.25em 0.5em;background:#4CAF50;color:white;border-radius:3px;text-decoration:none">{label}</a>'
        return href
    def select_file_preview_true(self, key):
        files_with_paths = self.list_all_files()        
        # Strip the directories from the file paths for display purposes
        files_display = [file_path.replace(d, '') for d in self.directories for file_path in files_with_paths if file_path.startswith(d)]
        # Add a "None" option as the first item in the list
        files_display.insert(0, "Select a file")

        if files_with_paths:
            # Select a file from the list with paths stripped for display
            # selected_file_display = st.selectbox("Select a file", files_display, index=0)
            selected_file_display = st.selectbox("Select a file", files_display, index=0, key=key)
            # Get the full path from the display selection if it's not "None"
            selected_file = next((f for f in files_with_paths if f.endswith(selected_file_display)), "") if selected_file_display != "None" else ""

            if selected_file:
                # Use an expander to show the file preview
                with st.expander("Preview File", expanded=True):
                    self.preview_file(selected_file)

                return selected_file
        else:
            st.error("No files found or there's an error accessing the directories.")

    def select_file_preview_false(self, key):
        files_with_paths = self.list_all_files()        
        # Strip the directories from the file paths for display purposes
        files_display = [file_path.replace(d, '') for d in self.directories for file_path in files_with_paths if file_path.startswith(d)]
        # Add a "Select a file" option as the first item in the list
        files_display.insert(0, "Select a file")

        if files_with_paths:
            # # Display the number of found files
            # st.write(f"Found {len(files_with_paths) - 1} files:")  # Subtract 1 for the "Select a file" option

            # Select a file from the list with paths stripped for display
            selected_file_display = st.selectbox("Select a file", files_display, index=0, key=key)

            # Get the full path from the display selection if it's not "Select a file"
            selected_file = next((f for f in files_with_paths if f.endswith(selected_file_display)), "") if selected_file_display != "Select a file" else ""

            if selected_file:
                # Use an expander to optionally show the file preview
                with st.expander("Preview File", expanded=False):
                    self.preview_file(selected_file)

                # Show download button
                st.markdown(self.file_download_button(selected_file), unsafe_allow_html=True)
                
            return selected_file  # This will return the selected file regardless of whether the expander is used
        else:
            st.error("No files found or there's an error accessing the directories.")
             
    def run(self):
        files_with_paths = self.list_all_files()        
        # Strip the directories from the file paths for display purposes
        files_display = [file_path.replace(d, '') for d in self.directories for file_path in files_with_paths if file_path.startswith(d)]
        # Add a "None" option as the first item in the list
        files_display.insert(0, "Select a file")
        st.write('# FILE MANAGEMENT SYSTEM')
        if files_with_paths:
            # st.write(f"Found {len(files_with_paths) - 1} files:")  # Subtract 1 for the "None" option
            # Select a file from the list with paths stripped for display
            selected_file_display = st.selectbox("Select a file", files_display, index=0)

            # Get the full path from the display selection if it's not "None"
            selected_file = next((f for f in files_with_paths if f.endswith(selected_file_display)), "") if selected_file_display != "None" else ""

            if selected_file:
                # Use an expander to show the file preview
                with st.expander("Preview File", expanded=False):
                    self.preview_file(selected_file)

                # Show download button
                st.markdown(self.file_download_button(selected_file), unsafe_allow_html=True)
                
                # Add button to delete the file
                if st.button(f"Delete {os.path.basename(selected_file)}"):
                    if self.delete_file(selected_file):
                        st.experimental_rerun()
        else:
            st.error("No files found or there's an error accessing the directories.")
    
def main():
    st.title("File Management App")
    file_manager = FileManager("/home/baio/data/output/", "/home/baio/data/upload/")
    file_manager.run()

# if __name__ == "__main__":
#     main()
