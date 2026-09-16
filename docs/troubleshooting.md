# Troubleshooting Guide

## 1. Ollama Connection Issues
- **Symptom:** UI displays "🔴 Ollama Server Offline"
- **Cause:** Ollama daemon is not running on port 11434.
- **Fix:** Start Ollama from CMD:
  ```cmd
  ollama serve
  ```
  Or launch the Ollama Windows taskbar application.

## 2. Model Not Found
- **Symptom:** "Target model 'qwen2.5:3b' not found"
- **Cause:** The model has not been downloaded locally.
- **Fix:** Pull model in CMD:
  ```cmd
  ollama pull qwen2.5:3b
  ```

## 3. FAISS Index Mismatch
- **Symptom:** Dimension assertion error during retrieval
- **Cause:** Changing the embedding model without deleting existing index storage.
- **Fix:** Remove storage folder:
  ```cmd
  rmdir /s /q storage
  ```
  The application will automatically recreate clean indexes on next ingestion.
