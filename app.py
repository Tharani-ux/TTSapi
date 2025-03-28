from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import easyocr
from PIL import Image
import numpy as np
import io
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR Text Extraction API",
    description="Extract text from images using EasyOCR",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# ✅ CORS Configuration (Fixed for Vite Frontend)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://0.0.0.0:8000",
    "http://localhost:5173"  # ✅ Allow Vite frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]  # ✅ Expose response headers
)

# ✅ Initialize EasyOCR model with error handling
try:
    logger.info("Loading EasyOCR model...")
    start_time = time.time()
    reader = easyocr.Reader(["en"])
    load_time = time.time() - start_time
    logger.info(f"EasyOCR model loaded successfully in {load_time:.2f} seconds")
except Exception as e:
    logger.error(f"Failed to load EasyOCR model: {str(e)}")
    raise RuntimeError("OCR model initialization failed")

# ✅ Constants
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

async def validate_image(file: UploadFile):
    """Validate image type and size"""
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": "unsupported_file_type", "allowed_types": list(ALLOWED_EXTENSIONS)}
        )

    file.file.seek(0, 2)
    file_size = file.file.tell()
    await file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": "file_too_large", "max_size_mb": MAX_FILE_SIZE // (1024 * 1024)}
        )

@app.post("/extract-text", summary="Extract text from an image")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from an uploaded image.
    - Max file size: 5MB
    - Supported formats: JPEG, PNG, WEBP
    - Returns extracted text
    """
    try:
        await validate_image(file)

        logger.info(f"Processing file: {file.filename} (Size: {file.size} bytes)")

        start_time = time.time()
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        ocr_start = time.time()
        result = reader.readtext(image_np, detail=0, paragraph=True)
        ocr_time = time.time() - ocr_start

        extracted_text = "\n".join(result) if result else "No text detected"
        process_time = time.time() - start_time

        logger.info(f"Processed {file.filename} in {process_time:.2f}s (OCR: {ocr_time:.2f}s)")

        return {
            "status": "success",
            "filename": file.filename,
            "text": extracted_text,
            "metrics": {
                "characters": len(extracted_text),
                "lines": len(extracted_text.split("\n")),
                "processing_time_sec": round(process_time, 2),
                "ocr_time_sec": round(ocr_time, 2)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "processing_error", "message": "Failed to process image"}
        )

@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "ocr-api",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render provides PORT as an environment variable
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
