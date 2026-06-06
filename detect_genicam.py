from harvesters.core import Harvester
from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("NewMyYolov12.pt")
h = Harvester()

print("add file C:\Program Files\iCentral\iCentral\Runtime\\x64\MVProducerGEV.cti to harvestor")
h.add_file("MVProducerGEV.cti")
h.update()

print("Found Camera:", h.device_info_list)

with h.create() as ia:
    
    try: 
        node_map = ia.remote_device.node_map
        node_map.BalanceWhiteAuto.value = "Continuous"
        print("Auto Whit Balance")
    except Exception as e:
        print(f'Could not set white balance {e}')

    ia.start()
    print("Camera Started")
    while True:
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            width = component.width
            height = component.height
            data = component.data.reshape(height, width)

            frame = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_BayerBG2BGR)

            results = model(frame, conf=0.5)
            annotated = results[0].plot()

            TRACK_CLASSES = ["half_ripe", "ripe", "overripe"]
            counts = {}
            cutting = {}

            #Counting per Class
            for box in results[0].boxes:
                label = model.names[int(box.cls)]
                if label in TRACK_CLASSES:
                    cutting[label] = cutting.get(label, 0) + 1
                counts[label] = counts.get(label, 0) + 1
                

            # Drawing with cv2
            total = sum(counts.values())
            cv2.putText(annotated, f"Total: {total}", (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            
            total_cut = sum(cutting.values())

            if total == 0:
                percentage = "Null"
            else:
                percentage = f"{(total_cut/total) * 100:.2f}%"

            cv2.putText(annotated, f"Cutting_Percentage: {percentage}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)

            cv2.imshow("YOLOv12 - ContrasTech Mars", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    h.reset()
