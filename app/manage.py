import os
import uvicorn

def main():
    os.environ.setdefault('FASTAPI_ENV', 'settings')
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError as exc:
        raise ImportError(
            "Couldn't import FastAPI. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

if __name__ == '__main__':
    main()