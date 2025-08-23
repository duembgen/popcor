# Copilot Instructions

YOU MUST UBSOLUTELY FOLLOW THESE INSTRUCTIONS AT ALL TIMES.

## General guidelines to be used throughout:
- Use spaces, not tabs: 4 spaces per indentation level 
- Variables are snake_case, Classes are CamelCase.

## When I ask you to overhaul a module, follow these instructions:
- Add type hints to all functions and methods. If a variable is initialized to None, make sure to use | None in the type hint.
- ABSOLUTELY do not document standard functions such as __init__, __str__, __repr__, unless there is something special to document.
- ABSOLUTELY do not document property funcitons (those that use @property decorator), unless there is something special to document.
- Never remove inline comments, those starting with #. They are important for understanding the code. You may correct or complete them.
- Add at least one-line docstrings to each function, unless the function starts with an underscore, and unless it is clear from the functon name what it does.
- ABSOLUTELY if the function or class already has a docstring, do not rewrite it, only correct or complete it.
- Fix typos in documentation. Do not make any unnecessary changes to language, only if something is grammatically incorrect. 
- Make code more efficient without making it less readable. 
- If the module contains mostly one class, just provide one very brief module-level docstring. All the main information should be in the class top-level docstring. 
- If this module contains many helper functions, then add a top-level docstring if it does not exist yet. If there is one, make sure that it features all important aspects of the code. Do not add information such as “Requires:, Author:, Date:”, only the technical content of the file. 
- Do not introduce dummy variables, like d=self.d inside a function. Always use self.d for better readability.