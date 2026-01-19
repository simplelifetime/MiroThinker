# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import asyncio
import json
import os
import shlex
from urllib.parse import urlparse

from e2b_code_interpreter import Sandbox
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("e2b-python-interpreter")

# API keys
E2B_API_KEY = os.environ.get("E2B_API_KEY")
LOGS_DIR = os.environ.get(
    "LOGS_DIR", "../../logs"
)  # Directory where benchmark logs are stored

# DEFAULT TEMPLATE ID
DEFAULT_TEMPLATE_ID = "1av7fdjfvcparqo8efq6"

# DEFAULT CONFS
DEFAULT_TIMEOUT = 600  # seconds
# Maximum number of tokens that can be returned by the Python tool
MAX_RESULT_LEN = 20_000
# Maximum number of tokens allowed in an error message
MAX_ERROR_LEN = 4_000
# Invalid sandbox IDs that are not allowed to be used
INVALID_SANDBOX_IDS = {
    "default",
    "sandbox1",
    "sandbox",
    "some_id",
    "new_sandbox",
    "python",
    "create_sandbox",
    "sandbox123",
    "temp",
    "sandbox-0",
    "sandbox-1",
    "sandbox_0",
    "sandbox_1",
    "new",
    "0",
    "auto",
    "default_sandbox",
    "none",
    "sandbox_12345",
    "dummy",
    "sandbox_01",
}


def looks_like_dir(path: str) -> bool:
    """
    Return True if the given path either:
      - exists and is a directory, OR
      - does not exist but looks like a directory (e.g., ends with '/', or has no file extension)
    """
    # If it exists, trust the filesystem
    if os.path.isdir(path):
        return True

    # If it ends with '/' or has no extension, treat as directory
    if path.endswith(os.path.sep) or not os.path.splitext(path)[1]:
        return True

    return False


def truncate_result(result: str) -> str:
    """
    Truncate result to MAX_RESULT_LEN.

    Args:
        result: The full result string to potentially truncate

    Returns:
        Truncated result string
    """
    if len(result) > MAX_RESULT_LEN:
        result = result[:MAX_RESULT_LEN] + " [Result truncated due to length limit]"

    return result


@mcp.tool()
async def create_sandbox(timeout: int = DEFAULT_TIMEOUT) -> str:
    """Create a linux sandbox.

    Args:
        timeout: Time in seconds before the sandbox is automatically shutdown. The default is 600 seconds.

    Returns:
        The sandbox_id of the newly created sandbox. You should use this sandbox_id to run other tools in the sandbox.
    """
    max_retries = 5
    timeout = min(timeout, DEFAULT_TIMEOUT)
    for attempt in range(1, max_retries + 1):
        sandbox = None
        try:
            sandbox = Sandbox(
                template=DEFAULT_TEMPLATE_ID,
                timeout=timeout,
                api_key=E2B_API_KEY,
            )
            info = sandbox.get_info()

            tmpfiles_dir = os.path.join(LOGS_DIR, "tmpfiles")
            os.makedirs(tmpfiles_dir, exist_ok=True)

            return f"Sandbox created with sandbox_id: {info.sandbox_id}"
        except Exception as e:
            if attempt == max_retries:
                error_details = str(e)[:MAX_ERROR_LEN]
                return f"[ERROR]: Failed to create sandbox after {max_retries} attempts: {error_details}, please retry later."
            await asyncio.sleep(attempt**2)  # Exponential backoff
        finally:
            # Set timeout before exit to prevent timeout after function exits
            try:
                sandbox.set_timeout(timeout)
            except Exception:
                pass  # Ignore timeout setting errors


@mcp.tool()
async def run_command(command: str, sandbox_id: str) -> str:
    """Execute a lightweight shell command in the linux sandbox (no long-running, blocking, or resource-heavy processes).

    Args:
        command: The command to execute.
        sandbox_id: The id of the sandbox to execute the command in. To create a new sandbox, use tool `create_sandbox`.

    Returns:
        A CommandResult object containing the result of the command execution, format like CommandResult(stderr=..., stdout=..., exit_code=..., error=...)
    """
    if sandbox_id in INVALID_SANDBOX_IDS:
        return f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a real sandbox first using the create_sandbox tool."

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}. Make sure the sandbox is created and the sandbox_id is correct."

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            sandbox.set_timeout(
                DEFAULT_TIMEOUT
            )  # refresh the timeout for each command execution
            result = sandbox.commands.run(command)

            result_str = str(result)
            return truncate_result(result_str)
        except Exception as e:
            if attempt == max_retries:
                # Build error message
                error_details = str(e)[:MAX_ERROR_LEN]
                error_msg = f"[ERROR]: Failed to run command after {max_retries} attempts.\n\nException type: {type(e).__name__}\nDetails: {error_details}"
                return error_msg
            await asyncio.sleep(attempt**2)  # Exponential backoff
        finally:
            # Set timeout before exit to prevent timeout after function exits
            try:
                sandbox.set_timeout(DEFAULT_TIMEOUT)
            except Exception:
                pass  # Ignore timeout setting errors


@mcp.tool()
async def run_python_code(code_block: str, sandbox_id: str) -> str:
    """Run short, safe python code in a sandbox and return the execution result (avoid long loops or heavy tasks; must finish quickly).

    Args:
        code_block: The python code to run.
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.

    Returns:
        A CommandResult object containing the result of the command execution, format like CommandResult(stderr=..., stdout=..., exit_code=..., error=...)
    """
    if sandbox_id in INVALID_SANDBOX_IDS:
        return f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a real sandbox first using the create_sandbox tool."

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}. Make sure the sandbox is created and the sandbox_id is correct."

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            sandbox.set_timeout(
                DEFAULT_TIMEOUT
            )  # refresh the timeout for each command execution

            execution = sandbox.run_code(code_block)
            result_str = str(execution)
            return truncate_result(result_str)
        except Exception as e:
            if attempt == max_retries:
                error_details = str(e)[:MAX_ERROR_LEN]
                error_msg = f"[ERROR]: Failed to run code in sandbox {sandbox_id} after {max_retries} attempts. Exception type: {type(e).__name__}, Details: {error_details}"
                return error_msg
            await asyncio.sleep(attempt**2)  # Exponential backoff
        finally:
            # Set timeout before exit to prevent timeout after function exits
            try:
                sandbox.set_timeout(DEFAULT_TIMEOUT)
            except Exception:
                pass  # Ignore timeout setting errors


@mcp.tool()
async def upload_file_from_local_to_sandbox(
    sandbox_id: str, local_file_path: str, sandbox_file_path: str = "/home/user"
) -> str:
    """Upload a local file to the `/home/user` dir of the remote python interpreter.

    Args:
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
        local_file_path: The path of the file on local machine to upload.
        sandbox_file_path: The path of directory to upload the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the uploaded file in the remote python interpreter if the upload is successful.
    """
    if sandbox_id in INVALID_SANDBOX_IDS:
        return f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a real sandbox first using the create_sandbox tool."

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}. Make sure the sandbox is created and the sandbox_id is correct."

    try:
        sandbox.set_timeout(
            DEFAULT_TIMEOUT
        )  # refresh the timeout for each command execution

        # Check if local file exists and is readable
        if not os.path.exists(local_file_path):
            return f"[ERROR]: Local file does not exist: {local_file_path}"
        if not os.path.isfile(local_file_path):
            return f"[ERROR]: Path is not a file: {local_file_path}"

        # Get the uploaded file path
        uploaded_file_path = os.path.join(
            sandbox_file_path, os.path.basename(local_file_path)
        )
        # Normalize the path
        uploaded_file_path = os.path.normpath(uploaded_file_path)

        # Ensure the parent directory exists in sandbox
        parent_dir = os.path.dirname(uploaded_file_path)
        if parent_dir and parent_dir != "/":
            mkdir_result = sandbox.commands.run(f"mkdir -p {shlex.quote(parent_dir)}")
            if mkdir_result.exit_code != 0:
                mkdir_result_str = str(mkdir_result)[:MAX_ERROR_LEN]
                return f"[ERROR]: Failed to create directory {parent_dir} in sandbox {sandbox_id}: {mkdir_result_str}"

        # Upload the file
        with open(local_file_path, "rb") as f:
            sandbox.files.write(uploaded_file_path, f)

        return f"File uploaded to {uploaded_file_path}"
    except Exception as e:
        error_details = str(e)[:MAX_ERROR_LEN]
        return f"[ERROR]: Failed to upload file {local_file_path} to sandbox {sandbox_id}: {error_details}"
    finally:
        # Set timeout before exit to prevent timeout after function exits
        try:
            sandbox.set_timeout(DEFAULT_TIMEOUT)
        except Exception:
            pass  # Ignore timeout setting errors


@mcp.tool()
async def download_file_from_internet_to_sandbox(
    sandbox_id: str, url: str, sandbox_file_path: str = "/home/user"
) -> str:
    """Download a file from the internet to the `/home/user` dir of the sandbox (avoid large or slow URLs).

    Args:
        sandbox_id: The id of the sandbox to run the code in. Reuse existing sandboxes whenever possible. To create a new sandbox, use tool `create_sandbox`.
        url: The URL of the file to download.
        sandbox_file_path: The path of directory to download the file to in the sandbox. Default is `/home/user/`.

    Returns:
        The path of the downloaded file in the sandbox if the download is successful.
    """
    if sandbox_id in INVALID_SANDBOX_IDS:
        return f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a real sandbox first using the create_sandbox tool."

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}. Make sure the sandbox is created and the sandbox_id is correct."

    try:
        sandbox.set_timeout(
            DEFAULT_TIMEOUT
        )  # refresh the timeout for each command execution

        # Extract basename from URL properly (handle query parameters)
        parsed_url = urlparse(url)
        basename = os.path.basename(parsed_url.path) or "downloaded_file"
        # Remove any query parameters or fragments from basename
        if "?" in basename:
            basename = basename.split("?")[0]
        if "#" in basename:
            basename = basename.split("#")[0]

        # Check whether sandbox_file_path looks like a directory
        if looks_like_dir(sandbox_file_path):
            # It's a directory — join with the filename
            downloaded_file_path = os.path.join(sandbox_file_path, basename)
        else:
            # It's a file path — use it directly
            downloaded_file_path = sandbox_file_path

        # Normalize the path
        downloaded_file_path = os.path.normpath(downloaded_file_path)

        # Ensure the parent directory exists in sandbox
        parent_dir = os.path.dirname(downloaded_file_path)
        if parent_dir and parent_dir != "/":
            mkdir_result = sandbox.commands.run(f"mkdir -p {shlex.quote(parent_dir)}")
            if mkdir_result.exit_code != 0:
                mkdir_result_str = str(mkdir_result)[:MAX_ERROR_LEN]
                return f"[ERROR]: Failed to create directory {parent_dir} in sandbox {sandbox_id}: {mkdir_result_str}"

        # Download the file with retry logic
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            safe_url = shlex.quote(url)
            safe_path = shlex.quote(downloaded_file_path)
            cmd = f"wget {safe_url} -O {safe_path}"
            try:
                result = sandbox.commands.run(cmd)
                if result.exit_code == 0:
                    return f"File downloaded to {safe_path}"
                elif attempt < max_retries:
                    await asyncio.sleep(4**attempt)
                    continue  # Retry
                else:
                    # Extract detailed error information
                    error_details = ""
                    if hasattr(result, "stderr") and result.stderr:
                        error_details = f"stderr: {result.stderr}"[:MAX_ERROR_LEN]
                    error_msg = (
                        f"[ERROR]: Failed to download file from {url} to {downloaded_file_path} after {max_retries} attempts.\n\n"
                        f"exit_code: {result.exit_code}\n\n"
                        f"Details: {error_details}"
                    )
                    return error_msg
            except Exception as e:
                if attempt == max_retries:
                    error_details = str(e)[:MAX_ERROR_LEN]
                    error_msg = f"[ERROR]: Failed to download file from {url} to {downloaded_file_path}. Exception: {error_details}"
                    return error_msg
                await asyncio.sleep(4**attempt)
    except Exception as e:
        error_details = str(e)[:MAX_ERROR_LEN]
        return f"[ERROR]: Failed to download file from {url}: {error_details}"
    finally:
        # Set timeout before exit to prevent timeout after function exits
        try:
            sandbox.set_timeout(DEFAULT_TIMEOUT)
        except Exception:
            pass  # Ignore timeout setting errors


@mcp.tool()
async def download_file_from_sandbox_to_local(
    sandbox_id: str, sandbox_file_path: str, local_filename: str = None
) -> str:
    """Download a file from the sandbox to local system. Files in sandbox cannot be processed by tools from other servers - only local files and internet URLs can be processed by them.

    Args:
        sandbox_id: The id of the sandbox to download the file from. To have a sandbox, use tool `create_sandbox`.
        sandbox_file_path: The path of the file to download on the sandbox.
        local_filename: Optional filename to save as. If not provided, uses the original filename from sandbox_file_path.

    Returns:
        The local path of the downloaded file if successful, otherwise error message.
    """
    if sandbox_id in INVALID_SANDBOX_IDS:
        return f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a real sandbox first using the create_sandbox tool."

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}. Make sure the sandbox is created and the sandbox_id is correct."

    try:
        sandbox.set_timeout(
            DEFAULT_TIMEOUT
        )  # refresh the timeout for each command execution

        # Create tmpfiles directory if it doesn't exist
        if not LOGS_DIR:
            return "[ERROR]: LOGS_DIR environment variable is not set. Cannot determine where to save the file."

        tmpfiles_dir = os.path.join(LOGS_DIR, "tmpfiles")
        os.makedirs(tmpfiles_dir, exist_ok=True)

        # Check if the path is a directory (before attempting to read)
        check_result = sandbox.commands.run(
            f'test -d {shlex.quote(sandbox_file_path)} && echo "is_directory" || echo "not_directory"'
        )
        if check_result.stdout and "is_directory" in check_result.stdout:
            return f"[ERROR]: Cannot download '{sandbox_file_path}' from sandbox {sandbox_id}: path is a directory, not a file."

        # Check if the file exists
        check_file_result = sandbox.commands.run(
            f'test -f {shlex.quote(sandbox_file_path)} && echo "exists" || echo "not_exists"'
        )
        if check_file_result.stdout and "not_exists" in check_file_result.stdout:
            # Check if it exists at all (might be a symlink or other type)
            check_any_result = sandbox.commands.run(
                f'test -e {shlex.quote(sandbox_file_path)} && echo "exists" || echo "not_exists"'
            )
            if check_any_result.stdout and "not_exists" in check_any_result.stdout:
                error_msg = f"[ERROR]: Cannot download '{sandbox_file_path}' from sandbox {sandbox_id}: file does not exist."
                return error_msg

        # Determine local filename
        if local_filename is None or local_filename.strip() == "":
            local_filename = os.path.basename(sandbox_file_path)
            # If basename is empty or just '/', use a default name
            if not local_filename or local_filename == "/":
                local_filename = "downloaded_file"

        local_file_path = os.path.join(
            tmpfiles_dir, f"sandbox_{sandbox_id}_{local_filename}"
        )

        # Download the file
        try:
            with open(local_file_path, "wb") as f:
                content = sandbox.files.read(sandbox_file_path, format="bytes")
                f.write(content)
        except Exception as read_error:
            error_msg = str(read_error).lower()
            if "directory" in error_msg or "is a directory" in error_msg:
                return f"[ERROR]: Cannot download '{sandbox_file_path}' from sandbox {sandbox_id}: path is a directory, not a file."
            else:
                read_error_details = str(read_error)[:MAX_ERROR_LEN]
                return f"[ERROR]: Failed to read file '{sandbox_file_path}' from sandbox {sandbox_id}: {read_error_details}"

        return f"File downloaded successfully to: {local_file_path}"
    except Exception as e:
        error_details = str(e)[:MAX_ERROR_LEN]
        return f"[ERROR]: Failed to download file '{sandbox_file_path}' from sandbox {sandbox_id}: {error_details}"
    finally:
        # Set timeout before exit to prevent timeout after function exits
        try:
            sandbox.set_timeout(DEFAULT_TIMEOUT)
        except Exception:
            pass  # Ignore timeout setting errors


@mcp.tool()
async def process_image_with_code(
    sandbox_id: str,
    code: str,
    image_data: str = None,
    image_url: str = None,
    image_format: str = "PNG",
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Process an image using Python code in the sandbox environment.

    This tool allows you to load an image (from base64 data or URL) into the sandbox,
    execute Python code to process it (resize, crop, rotate, filter, etc.),
    and return the processed image as base64 data.

    Args:
        sandbox_id: The id of the sandbox to run the code in. To create a new sandbox, use tool `create_sandbox`.
        code: Python code to execute for image processing. The code should work with a PIL Image object named 'image_1'.
        image_data: Base64 encoded image data to load into the sandbox.
        image_url: URL of image to download and load into the sandbox (alternative to image_data).
        image_format: Output image format (PNG, JPEG, etc.) for the processed result.
        timeout: Time in seconds before the execution times out.

    Returns:
        JSON string containing the execution result, including processed image data if successful.
    """
    if sandbox_id in INVALID_SANDBOX_IDS:
        return f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a real sandbox first using the create_sandbox tool."

    # Validate input parameters
    if not image_data and not image_url:
        return "[ERROR]: Either 'image_data' or 'image_url' must be provided."

    if image_data and image_url:
        return "[ERROR]: Provide either 'image_data' or 'image_url', not both."

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=E2B_API_KEY)
    except Exception:
        return f"[ERROR]: Failed to connect to sandbox {sandbox_id}. Make sure the sandbox is created and the sandbox_id is correct."

    try:
        sandbox.set_timeout(min(timeout, DEFAULT_TIMEOUT))

        # Prepare initialization code for image loading
        init_code = """
from PIL import Image
import base64
from io import BytesIO
import requests

def pil_image_to_base64(img, format="PNG"):
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def base64_to_pil_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image
"""

        # Load image based on input type
        if image_data:
            # Load from base64 data
            init_code += f"""
# Load image from base64 data
_img_base64 = "{image_data}"
image_1 = base64_to_pil_image(_img_base64)
"""
        elif image_url:
            # Download and load from URL
            init_code += f"""
# Download and load image from URL
response = requests.get("{image_url}")
response.raise_for_status()
image_1 = Image.open(BytesIO(response.content))
"""

        # Combine initialization and user code
        full_code = init_code + "\n" + code + "\n"

        # Execute the code
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                execution = sandbox.run_code(full_code, timeout=timeout)

                if execution.error:
                    if attempt == max_retries:
                        error_details = str(execution.error)[:MAX_ERROR_LEN]
                        return f"[ERROR]: Code execution failed after {max_retries} attempts: {error_details}"
                    await asyncio.sleep(attempt**2)
                    continue

                # Get the stdout and stderr
                stdout = execution.logs.stdout if execution.logs else ""
                stderr = execution.logs.stderr if execution.logs else ""

                # Check if image_1 was modified and convert back to base64
                result_code = """
try:
    # Convert processed image back to base64
    processed_image_b64 = pil_image_to_base64(image_1, format='PNG')
    print(f"IMAGE_RESULT:{{processed_image_b64}}")
except Exception as e:
    print(f"IMAGE_CONVERSION_ERROR:{{str(e)}}")
"""

                result_execution = sandbox.run_code(result_code, timeout=30)
                result_stdout = result_execution.logs.stdout if result_execution.logs else ""
                result_stderr = result_execution.logs.stderr if result_execution.logs else ""

                # Extract image result from stdout
                processed_image_data = None
                if "IMAGE_RESULT:" in result_stdout:
                    # Extract base64 data between markers
                    start_marker = "IMAGE_RESULT:"
                    end_marker = "}"
                    start_idx = result_stdout.find(start_marker)
                    if start_idx != -1:
                        end_idx = result_stdout.find(end_marker, start_idx + len(start_marker))
                        if end_idx != -1:
                            processed_image_data = result_stdout[start_idx + len(start_marker):end_idx + 1]

                # Prepare response
                response_data = {
                    "success": processed_image_data is not None,
                    "stdout": truncate_result(stdout),
                    "stderr": truncate_result(stderr),
                    "result_stdout": truncate_result(result_stdout),
                    "result_stderr": truncate_result(result_stderr),
                }

                if processed_image_data:
                    response_data["processed_image_data"] = processed_image_data
                    response_data["image_format"] = image_format

                return json.dumps(response_data, ensure_ascii=False)

            except Exception as e:
                if attempt == max_retries:
                    error_details = str(e)[:MAX_ERROR_LEN]
                    return f"[ERROR]: Failed to process image after {max_retries} attempts: {error_details}"
                await asyncio.sleep(attempt**2)

    except Exception as e:
        error_details = str(e)[:MAX_ERROR_LEN]
        return f"[ERROR]: Failed to process image: {error_details}"
    finally:
        # Set timeout before exit
        try:
            sandbox.set_timeout(DEFAULT_TIMEOUT)
        except Exception:
            pass


if __name__ == "__main__":
    mcp.run(transport="stdio")
