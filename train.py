from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO("yolov8n-seg.pt")
    
 
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        workers=2,   
        device=0     
    )