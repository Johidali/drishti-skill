# DRISTHI AI (Vigil-Skill)
Run: `docker compose up --build` -> dashboard http://localhost:3000, API docs http://localhost:8000/docs
Accuracy: `pip install opencv-python-headless numpy && python evaluate_metrics.py --n 200`
Real YOLOv8: `pip install ultralytics`, then `edge_agent.py --detector yolo --weights <fine-tuned.pt>` (lathe/sewing need a custom-trained class).
