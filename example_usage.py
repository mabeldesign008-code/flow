"""
Example usage of the enhanced AudioRecorder with WhisprFlow integration.
"""
import time
from recorder import AudioRecorder


def example_basic_usage():
    """Basic usage - backward compatible with original code."""
    print("=== Basic Usage Example ===\n")
    
    recorder = AudioRecorder()
    
    print("Starting recording...")
    recorder.start()
    
    # Simulate recording for 2 seconds
    for i in range(20):
        level = recorder.get_level()
        print(f"Level: {level:.4f}", end='\r')
        time.sleep(0.1)
    
    print("\nStopping recording...")
    result = recorder.stop()
    
    if result:
        audio, sample_rate = result
        print(f"✓ Recorded {len(audio)/sample_rate:.2f} seconds at {sample_rate}Hz")
    else:
        print("✗ No audio recorded")


def example_device_selection():
    """Example of listing and selecting audio devices."""
    print("\n=== Device Selection Example ===\n")
    
    # List available devices
    devices = AudioRecorder.list_devices()
    print(f"Found {len(devices)} input device(s):\n")
    
    for device in devices:
        default_marker = " [DEFAULT]" if device.is_default else ""
        print(f"  {device.index}: {device.name}{default_marker}")
        print(f"     Channels: {device.channels}, Sample Rate: {device.sample_rate}Hz")
    
    if devices:
        # Use first device
        print(f"\nUsing device: {devices[0].name}")
        recorder = AudioRecorder(device=devices[0].index)
        print(f"✓ Recorder initialized with device {devices[0].index}")


def example_advanced_features():
    """Example using advanced features."""
    print("\n=== Advanced Features Example ===\n")
    
    # Create recorder with all enhancements
    recorder = AudioRecorder(
        sample_rate=None,              # Auto-detect
        adaptive_threshold=True,        # Adapt to noise
        noise_reduction=True,           # Remove noise
        min_duration=0.5,              # Require 0.5s minimum
        channels=1                      # Mono
    )
    
    print("Recorder configuration:")
    print(f"  Sample Rate: {recorder.sample_rate}Hz")
    print(f"  Channels: {recorder.channels}")
    print(f"  Adaptive Threshold: {recorder.adaptive_threshold_enabled}")
    print(f"  Noise Reduction: {recorder.noise_reduction_enabled}")
    
    # Start recording
    if recorder.start():
        print("\n✓ Recording started")
        
        # Monitor for 3 seconds
        for i in range(30):
            level = recorder.get_level()
            threshold = recorder.get_adaptive_threshold()
            print(f"Level: {level:.4f} | Threshold: {threshold:.4f}", end='\r')
            time.sleep(0.1)
        
        # Stop and get statistics
        result = recorder.stop()
        
        print("\n\nRecording Statistics:")
        stats = recorder.get_stats()
        print(f"  Frames Processed: {stats['frames_processed']}")
        print(f"  Error Count: {stats['error_count']}")
        print(f"  Buffer Overflows: {stats['buffer_overflows']}")
        print(f"  Clipping Detected: {stats['clipping_detected']}")
        print(f"  Final Threshold: {stats['adaptive_threshold']:.4f}")
        
        if result:
            audio, sr = result
            duration = len(audio) / sr
            print(f"\n✓ Successfully recorded {duration:.2f} seconds")
        else:
            print("\n✗ Recording failed or too short")
    else:
        print("✗ Failed to start recording")


def example_error_handling():
    """Example of proper error handling."""
    print("\n=== Error Handling Example ===\n")
    
    try:
        # Try to use invalid device
        recorder = AudioRecorder(device=999)
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Create valid recorder
    recorder = AudioRecorder()
    
    # Try to stop without starting
    result = recorder.stop()
    if result is None:
        print("✓ Correctly handled stop-without-start")
    
    # Start recording
    if recorder.start():
        print("✓ Recording started")
        
        # Try to start again (should fail gracefully)
        if not recorder.start():
            print("✓ Correctly prevented double-start")
        
        # Check state
        if recorder.is_recording():
            print("✓ Recording state is correct")
        
        recorder.stop()


def example_whisprflow_integration():
    """Example showing integration with WhisprFlow app."""
    print("\n=== WhisprFlow Integration Example ===\n")
    
    # This is how WhisprFlow can use the enhanced recorder
    recorder = AudioRecorder(
        adaptive_threshold=True,
        noise_reduction=True,
    )
    
    print("WhisprFlow-style usage:")
    print("1. User presses hotkey")
    
    if recorder.start():
        print("2. Recording started (visual feedback shown)")
        
        # Simulate user speaking for 2 seconds
        print("3. User speaks...")
        for i in range(20):
            level = recorder.get_level()
            # In real app, this would update the overlay visualization
            print(f"   Audio level: {'█' * int(level * 50)}", end='\r')
            time.sleep(0.1)
        
        print("\n4. User releases hotkey")
        result = recorder.stop()
        
        if result:
            audio, sr = result
            print(f"5. Processing {len(audio)/sr:.2f}s of audio...")
            
            # Check for issues
            stats = recorder.get_stats()
            if stats['clipping_detected']:
                print("   ⚠ Warning: Clipping detected - microphone too loud")
            if stats['buffer_overflows'] > 0:
                print(f"   ⚠ Warning: {stats['buffer_overflows']} buffer overflows")
            
            print("6. Audio ready for transcription ✓")
        else:
            print("5. No valid audio captured ✗")


if __name__ == "__main__":
    print("Enhanced AudioRecorder Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_device_selection()
        example_advanced_features()
        example_error_handling()
        example_whisprflow_integration()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()
