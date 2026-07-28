"""
Test Screen Context Capture
Quick test to verify OCR is working and see performance.
"""
import time
from screen_context import ScreenContext


def main():
    print("=" * 70)
    print("Screen Context Test")
    print("=" * 70)
    
    context = ScreenContext()
    
    if not context.enabled:
        print("\n❌ No OCR engine available!")
        print("\nInstall one:")
        print("  pip install paddlepaddle paddleocr  (fastest, recommended)")
        print("  pip install easyocr  (slower fallback)")
        return
    
    print(f"\n✓ Using {context._engine.upper()} OCR engine")
    print("\nInstructions:")
    print("1. Open any application with text (browser, editor, etc.)")
    print("2. This will capture screen text every 3 seconds")
    print("3. Press Ctrl+C to stop\n")
    
    if context._engine == "paddle":
        print("⏳ Loading PaddleOCR model (first time ~10 seconds)...\n")
    else:
        print("⏳ Loading EasyOCR model (first time ~30 seconds)...\n")
    
    try:
        count = 0
        while True:
            count += 1
            print(f"\n{'─' * 70}")
            print(f"Capture #{count} - {time.strftime('%H:%M:%S')}")
            print("─" * 70)
            
            start = time.time()
            text = context.get_screen_text(use_cache=False)
            elapsed = time.time() - start
            
            word_count = len(text.split()) if text else 0
            
            print(f"⏱️  Processing time: {elapsed:.2f}s")
            print(f"📝 Words extracted: {word_count}")
            print(f"\nFirst 300 characters:")
            print(text[:300] if text else "(no text detected)")
            
            if word_count > 0:
                print(f"\n✓ Successfully captured {word_count} words")
                
                # Show what would be sent to LLM
                llm_context = context.get_context_for_llm(max_words=150)
                print(f"\nLLM Context (limited to 150 words):")
                print(llm_context[:200] + "..." if len(llm_context) > 200 else llm_context)
            else:
                print("\n⚠️  No text detected")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\n✓ Test stopped")


if __name__ == "__main__":
    main()
