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