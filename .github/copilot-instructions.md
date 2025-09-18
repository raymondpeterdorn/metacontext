### GitHub Copilot Instructions

This document outlines the rules and best practices for using GitHub Copilot in this project. By following these guidelines, we can ensure consistent, high-quality, and maintainable code.

---

### **I. Core Philosophy & Methodology**

* **Project Philosophy:** Follow the "make it work -> make it right -> make it better" principle. The priority is to ship a correct Minimum Viable Product (MVP), then refactor for clarity, and finally optimize for performance.
* **Python Target:** All code and tooling should target **Python 3.12**.
* **Generics:** Prefer built-in generics (**`list`**, **`dict`**, **`tuple`**) over their counterparts from the `typing` module (e.g., `typing.List`).
* **Modular Design:** The project structure must be modular and layered. Each logical module should have isolated functionality and a stable public interface. Avoid tight coupling and maintain clear boundaries.

---

### **II. Code Quality & Static Analysis**

* **Type Annotations:**
    * **All public functions and classes must be fully type annotated.**
    * Enable strict type checking with **Mypy**.
    * If a MyPy warning or error is not resolved after **three attempts**, move on to the next task.
* **Linting:**
    * Use **Ruff** as the default linter.
    * Adhere to the specific Ruff configurations defined in `pyproject.toml`.
    * Fix violations proactively. If a Ruff error is not resolved after **three attempts**, move on.
* **Style and Conventions:**
    * Follow **PEP 8**: `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
    * Keep functions small (<= **20 lines**) and cyclomatic complexity low (<= **10**) when feasible.
    * Avoid unexplained acronyms. If one must be used, define it near its first use.
    * Prefer **`pathlib.Path`** over the `os.path` module for file system operations.
    * Use **`Protocol`** for dependency inversion.
* **Organization:**
    * Use **one class per file** for complex classes.
    * Group related functions within modules.
    * Use `__init__.py` files to expose the public API of a module.
    * Prefix private functions and classes with an underscore (`_`).
    * All imports should be at the top of the file, grouped into standard library, third-party, and local imports, with a blank line between each group.

---

### **III. Data Handling & API Design**

* **Validation & Serialization:**
    * Use **Pydantic 2** for all external data validation and serialization.
    * Prefer `pydantic.BaseModel` with explicit configurations: `model_config = {'extra': 'forbid', 'validate_assignment': True}`.
    * Utilize `@field_validator` for custom validation logic.
    * Avoid using unclear `Annotated[...]` validator lambdas unless readability is significantly improved.
* **API Design:**
    * Use Pydantic models for both API request and response validation.
    * Follow **RESTful conventions** for defining endpoints.
    * Ensure the API has comprehensive **OpenAPI documentation**.
    * Use **dependency injection** to provide shared resources to API endpoints.

---

### **IV. Logging & Error Handling**

* **Logging Standards:**
    * Use **`structlog`** for structured logging.
    * Log with context-rich messages (e.g., `logger.info("User created", user_id=user.id, email=user.email)`).
    * **Do not use `print()` for debugging.** Always use the logging system.
    * Ensure sensitive information, especially environment variables, is never logged.
* **Exception Handling:**
    * Use **specific exceptions** instead of the generic `Exception`.
    * Create **custom exceptions** for business logic errors.
    * Use **`pydantic.ValidationError`** for data validation failures.
    * Log exceptions with context *before* raising them.

---

### **V. Dependencies & Configuration**

* **Dependency Management:** Use **Poetry** for managing project dependencies.
* **Configuration:**
    * Use environment variables for configuration.
    * Environment variables should be defined in a local `.env` file.
    * Provide a `.envrc` file with examples, but **do not commit actual secrets**.
    * Environments: Maintain `development`, `staging`, and `production`. The `staging` environment should be as similar to `production` as possible.

---

### **VI. Documentation & Testing**

* **Documentation:**
    * **Always read** `README.md`, `docs/`, and any other markdown files at the start of a conversation to gather project context.
    * Maintain and update the **`docs/`** directory whenever code changes affect behavior or operations.
* **Testing:**
    * Use **pytest**.
    * Write tests that are **deterministic, isolated, and fast**.
    * **You can add more stuff here, Gemini!** (e.g., generate unit tests for new functions, prioritize testing edge cases, use fixtures, write descriptive test names).

### **VII. Usage of Terminal Commands**

* **Code-Workspace**
    * This repository uses a code-workspace format. Each workspace has its own virtual environment already installed. Activate the virtual environment for the workspace you are working in before running any commands.