# COCO Folder dataset format:
    dataset_dir =>
        train ->
            images -> list of images
            labels.json (In array "categories" -> Label index 0 is reserved for the 'background' class.) 
        valid ->
            images
            labels.json
        test ->
            images
            labels.json


    DetectionResult(
    detections=[
            Detection(
                bounding_box=BoundingBox(origin_x=87, origin_y=283, width=77, height=123),
                categories=[
                    Category(
                        index=None,
                        score=0.5, 
                        display_name=None, 
                        category_name='person'
                    )
                ],
                keypoints=[]
            ),
            Detection(
                bounding_box=BoundingBox(origin_x=0, origin_y=282, width=88, height=141),
                categories=[
                    Category(
                        index=None,
                        score=0.4453125, 
                        display_name=None, 
                        category_name='person'
                    )
                ],
                keypoints=[]
            ) 
        ]
    )


