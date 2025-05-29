import os
import re
import logging
from typing import List, Set, Optional

# --- Constants ---
# Absolute path of the directory where the script is located
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Configuration for folder exclusion
EXCLUDED_FOLDERS: Set[str] = {".git", ".github", ".vscode", "__pycache__", "scripts"}
# Items (files or specific folders) to exclude when listing directory contents for projects
EXCLUDED_ITEMS_IN_LIST_DIR: Set[str] = {"README.md", "update_readme_folders.py", ".DS_Store"}

# README configuration
README_FILENAME: str = "README.md"
START_MARKER: str = "<!-- FOLDER_LIST_START -->"
END_MARKER: str = "<!-- FOLDER_LIST_END -->"
DEFAULT_README_TEMPLATE: str = f"""# AI Learning Journey

## Agenda

The agenda is to learn AI by practice and pushing all the learning here.

## Project Structure

Each folder in this repository represents a specific task or concept related to AI. Inside each folder, you will find:

*   A `README.md` file containing details about the task/concept and steps to run any associated code.
*   Relevant code, notebooks, or other files.

## Projects
{START_MARKER}
{END_MARKER}

"""

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Functions ---
def get_project_folders(root_dir: str = SCRIPT_DIR) -> List[str]:
    """
    Scans the given root directory and returns a sorted list of project folder names.

    Project folders are directories within root_dir that are not explicitly excluded.

    Args:
        root_dir: The absolute path to the directory to scan.
                  Defaults to the directory where this script is located.

    Returns:
        A sorted list of strings, where each string is a project folder name.
    """
    folders: List[str] = []
    try:
        for item_name in os.listdir(root_dir):
            if item_name in EXCLUDED_ITEMS_IN_LIST_DIR or item_name in EXCLUDED_FOLDERS:
                continue

            item_path: str = os.path.join(root_dir, item_name)
            if os.path.isdir(item_path):
                folders.append(item_name)
    except FileNotFoundError:
        logger.error(f"Root directory not found: {root_dir}")
        return []
    except PermissionError:
        logger.error(f"Permission denied to access directory: {root_dir}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while listing folders in {root_dir}: {e}")
        return []
        
    return sorted(folders)

def generate_markdown_folder_list(project_folders: List[str]) -> str:
    """
    Generates a markdown-formatted list of project folders.

    Args:
        project_folders: A list of project folder names.

    Returns:
        A string containing the markdown list.
    """
    if not project_folders:
        return "No projects found yet."
    
    markdown_items: List[str] = [f"*   [{folder}](./{folder}/)" for folder in project_folders]
    return "\n".join(markdown_items)

def update_readme_content(
    current_content: str,
    folder_list_markdown: str
) -> Optional[str]:
    """
    Updates the README content with the new folder list between specified markers.

    If markers are not found, it attempts to add them under a '## Projects' header
    or at the end of the file.

    Args:
        current_content: The current string content of the README file.
        folder_list_markdown: The markdown string of the folder list to insert.

    Returns:
        The updated README content as a string, or None if markers are missing and
        cannot be added appropriately.
    """
    content_with_markers = current_content
    
    # Ensure markers are present
    if START_MARKER not in content_with_markers or END_MARKER not in content_with_markers:
        projects_header = "## Projects"
        if projects_header in content_with_markers:
            # Insert markers after the "## Projects" header
            replacement_target = f"{projects_header}\n{START_MARKER}\n{folder_list_markdown}\n{END_MARKER}"
            content_with_markers = content_with_markers.replace(projects_header, replacement_target, 1)
            logger.info("Added markers under '## Projects' header in README.")
        else:
            # Append "## Projects" header and markers if header is not found
            content_with_markers += f"\n{projects_header}\n{START_MARKER}\n{folder_list_markdown}\n{END_MARKER}\n"
            logger.info("Appended '## Projects' header and markers to README.")
        return content_with_markers # Return content with newly added markers and list

    # Replace the content between existing markers
    pattern = re.compile(f"({re.escape(START_MARKER)})(.*?)({re.escape(END_MARKER)})", re.DOTALL)
    
    def replace_match_content(match: re.Match) -> str:
        return f"{match.group(1)}\n{folder_list_markdown}\n{match.group(3)}"

    new_readme_content, num_replacements = pattern.subn(replace_match_content, content_with_markers)

    if num_replacements == 0:
        # This case should be rare if marker adding logic above works
        logger.warning(
            f"Could not find markers '{START_MARKER}' and '{END_MARKER}' in README.md "
            "to update the folder list, even after attempting to add them. "
            "Appending list to the end."
        )
        # Fallback: append if markers somehow still not processed
        new_readme_content += f"\n{START_MARKER}\n{folder_list_markdown}\n{END_MARKER}\n"
        
    return new_readme_content

def manage_readme_file(
    readme_filepath: str = os.path.join(SCRIPT_DIR, README_FILENAME)
) -> None:
    """
    Manages the README file: reads it, updates folder list, and writes back.

    If the README file doesn't exist, it creates one using a default template.

    Args:
        readme_filepath: The absolute path to the README file.
                         Defaults to 'README.md' in the script's directory.
    """
    project_folders: List[str] = get_project_folders(root_dir=SCRIPT_DIR)
    folder_list_md: str = generate_markdown_folder_list(project_folders)
    
    current_readme_content: str
    try:
        with open(readme_filepath, "r", encoding="utf-8") as f:
            current_readme_content = f.read()
    except FileNotFoundError:
        logger.warning(f"{readme_filepath} not found. Creating a new one with default template.")
        current_readme_content = DEFAULT_README_TEMPLATE # This template already includes markers
        # No need to call update_readme_content separately for new file if template has markers
        # and list is empty initially or handled by template structure.
        # For simplicity, let update_readme_content handle insertion into the new template.

    updated_content: Optional[str] = update_readme_content(current_readme_content, folder_list_md)

    if updated_content is None:
        logger.error("Failed to update README content. Aborting write.")
        return

    try:
        with open(readme_filepath, "w", encoding="utf-8") as f:
            f.write(updated_content)
        logger.info(f"Successfully updated {readme_filepath} with project folders.")
    except IOError as e:
        logger.error(f"Failed to write updated content to {readme_filepath}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing to {readme_filepath}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    manage_readme_file()
