import os

import pooch

CACHE_DIR = os.path.join(pooch.os_cache(""), "c4data")

image_register = pooch.create(
    path=CACHE_DIR,
    base_url="",
    registry={
        "planes.png": "9bd39926e073a9092847803a18dfcc47a5a030a9e9cdb0f0e6ec2b7bc4479481",
        "triangle.png": "d2da7a29c269b05ab77ec77044ce36d3866d207cea0032af5ab7e8fc8e7f0580",
    },
    urls={
        "planes.png": "https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/images/planes.png",
        "triangle.png": "https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/images/triangle.png",
    },
)

video_register = pooch.create(
    path=CACHE_DIR,
    base_url="",
    registry={
        "aerobatics.mp4": "4ca78c67dc199793f975caf4fca0039958d290b9af8ae46b0f87863181336afe",
        "drifting_car.mp4": "c490de3c27ade26915b20a8130c9dd58d6a45c7152ea92cd059d31c4a5c370ec",
    },
    urls={
        "aerobatics.mp4": "https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/videos/aerobatics.mp4",
        "drifting_car.mp4": "https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/videos/drifting_car.mp4",
    },
)

nn_register = pooch.create(
    path=CACHE_DIR,
    base_url="",
    registry={
        # 'yolov3.weights': '523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297'
        "yolov3.weights": "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"
    },
    urls={
        # 'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights'
        "yolov3.weights": "https://huggingface.co/spaces/Epitech/Scarecrow/resolve/main/yolov3.weights"
    },
    retry_if_failed=1,
)

d3_f16_register = pooch.create(
    path=os.path.join(CACHE_DIR, "f16"),
    base_url="",
    registry={
        "Aileron_A_F16.stl": "43cdd5ef230eac34e9dd3aa727eed25500eeb4ab3733dc480c1c6b6be8ecc60f",
        "Aileron_B_F16.stl": "b59da3a237b10da2991af7850b033a584177b47eb1405928ca6a785bb6dc3935",
        "Body_F16.stl": "9f1a4de66fba156079c3049756b5f1aaaeb23250a6c90f6bf566ccc693d8baa0",
        "Cockpit_F16.stl": "d6e5fdca6e645df42e6b417346c313a8981d6c5c468362bf05f7bb607e719627",
        "LE_Slat_A_F16.stl": "dbfb18d5dccf8ee56c4a198b425f6569e23ed09a6c993f9fd97306e5bac12f35",
        "LE_Slat_B_F16.stl": "4a38c338767e671f09ac1c180f8983390c4441fb29a9398a6d9ba49092db1ae6",
        "Rudder_F16.stl": "8315ca56856cf9efd16937b738eeb436ccd566071d0817d85d5f943be6632769",
        "Stabilator_A_F16.stl": "a8074ddb0deea10cf5f764a81f68ea3b792201dd14b303b9a1c620cde88f6201",
        "Stabilator_B_F16.stl": "0c1192f5244f073086233466845350b735f6e9fce04a44a77fa93799754e6aec",
    },
    urls={
        "Aileron_A_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/Aileron_A_F16.stl",
        "Aileron_B_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/Aileron_B_F16.stl",
        "Body_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/Body_F16.stl",
        "Cockpit_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/Cockpit_F16.stl",
        "LE_Slat_A_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/LE_Slat_A_F16.stl",
        "LE_Slat_B_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/LE_Slat_B_F16.stl",
        "Rudder_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/Rudder_F16.stl",
        "Stabilator_A_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/Stabilator_A_F16.stl",
        "Stabilator_B_F16.stl": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/f16/Stabilator_B_F16.stl",
    },
)

d3_register = pooch.create(
    path=CACHE_DIR,
    base_url="",
    registry={
        "bunny.pcd": "64e10d06e9b9f9b7e4a728b0eff399334abe017cbf168e6c0ff39f01a025acc9",
        "bunny_mesh.ply": "b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe",
    },
    urls={
        "bunny.pcd": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/bunny.pcd",
        "bunny_mesh.ply": r"https://raw.githubusercontent.com/c4dynamics/c4dynamics/main/datasets/d3_models/bunny_mesh.ply",
    },
)
