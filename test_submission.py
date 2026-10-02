import time
import sys

submission_dir = r"C:\Users\rajcr\Desktop\Study\MTech\Courses\CV\Project\submission\MT2025721"

sys.path.insert(0, submission_dir)
if "solution" in sys.modules:
    del sys.modules["solution"]

from solution import TrafficViolationDetector

model = TrafficViolationDetector(model_dir="./models")

# First call (slow — warmup)
t = time.time()
out1 = model.predict("test_image.jpg")
print(f"First call: {time.time()-t:.1f}s")

# Second call (should be fast)
t = time.time()
out2 = model.predict("test_image.jpg")
print(f"Second call: {time.time()-t:.1f}s")