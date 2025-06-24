#!/usr/bin/env python3
"""
Test script to verify implicit expression results work with streaming output.

This tests that expressions like `raw_datasets["train"][0]` are properly
displayed in real-time just like explicit print statements.
"""

def test_implicit_expressions():
    """Test that implicit expressions are streamed properly."""
    print("=== Testing Implicit Expression Results ===")
    print()
    
    # Test 1: Simple expression result
    print("Test 1: Simple expression result")
    42  # This should be displayed as [Rank N] 42
    
    print()
    
    # Test 2: Dictionary access (simulating dataset access)
    print("Test 2: Dictionary access (simulating dataset access)")
    sample_data = {
        "train": [
            {"text": "Hello world", "label": 1},
            {"text": "Goodbye world", "label": 0}
        ]
    }
    sample_data["train"][0]  # This should be displayed
    
    print()
    
    # Test 3: Mixed print and implicit result
    print("Test 3: Mixed print and implicit result")
    print("About to show the result:")
    {"result": "success", "value": 123}  # This should be displayed
    
    print()
    
    # Test 4: Complex expression
    print("Test 4: Complex expression")
    [i**2 for i in range(5)]  # This should be displayed
    
    print()
    
    # Test 5: String expression
    print("Test 5: String expression")
    "This is an implicit string result"  # This should be displayed
    
    print()
    
    # Test 6: None result (should not be displayed)
    print("Test 6: None result (should not be displayed)")
    print("This print should appear, but None result should not:")
    None  # This should NOT be displayed (None results are suppressed)
    
    print("Test completed!")

if __name__ == "__main__":
    test_implicit_expressions() 