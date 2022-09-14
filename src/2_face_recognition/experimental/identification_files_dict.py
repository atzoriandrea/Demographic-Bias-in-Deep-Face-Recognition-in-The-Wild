model_name = "MobileFaceNet+NPCFace"
# Model_Dataset
data = {
    "DiveFace": {
        "HR_HR": {
            "npy": "/media/Workspace/Datasets/DiveFace/DatasetMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/DiveFace/old_data/DiveFace_resized.json",
            "testfile": "/media/Workspace/Datasets/DiveFace/testHR.txt"

        },
        "HR_LR": {
            "npy": "/media/Workspace/Datasets/DiveFace/WildDatasetMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/DiveFace/old_data/DiveFace_resized.json",
            "testfile": "/media/Workspace/Datasets/DiveFace/testLR.txt"

        },
        "LR_HR": {
            "npy": "/media/Workspace/Datasets/DiveFace/DatasetMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/DiveFace/old_data/DiveFace_resized.json",
            "testfile": "/media/Workspace/Datasets/DiveFace/testHR.txt"

        },
        "LR_LR": {
            "npy": "/media/Workspace/Datasets/DiveFace/WildDatasetMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/DiveFace/old_data/DiveFace_resized.json",
            "testfile": "/media/Workspace/Datasets/DiveFace/testLR.txt"

        }
    },
    "RFW": {
        "HR_HR": {
            "npy": "/media/Workspace/Datasets/RFW/DatasetMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/RFW/test_HR.json",
            "testfile": "/media/Workspace/Datasets/RFW/testHR.txt"

        },
        "HR_LR": {
            "npy": "/media/Workspace/Datasets/RFW/DatasetLRMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/RFW/test_LR.json",
            "testfile": "/media/Workspace/Datasets/RFW/testLR.txt"

        },
        "LR_HR": {
            "npy": "/media/Workspace/Datasets/RFW/DatasetMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/RFW/test_HR.json",
            "testfile": "/media/Workspace/Datasets/RFW/testHR.txt"

        },
        "LR_LR": {
            "npy": "/media/Workspace/Datasets/RFW/DatasetLRMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/RFW/test_LR.json",
            "testfile": "/media/Workspace/Datasets/RFW/testLR.txt"

        }
    },
    "CelabA": {
        "HR_HR": {
            "npy": "/media/Workspace/Datasets/CelebA/Aligned_DatasetMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/CelebA/test_HR.json",
            "testfile": "/media/Workspace/Datasets/CelebA/testHR.txt"

        },
        "HR_LR": {
            "npy": "/media/Workspace/Datasets/CelebA/WildAligned_DatasetMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/CelebA/test_LR.json",
            "testfile": "/media/Workspace/Datasets/CelebA/testLR.txt"

        },
        "LR_HR": {
            "npy": "/media/Workspace/Datasets/CelebA/Aligned_DatasetMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/CelebA/test_HR.json",
            "testfile": "/media/Workspace/Datasets/CelebA/testHR.txt"

        },
        "LR_LR": {
            "npy": "/media/Workspace/Datasets/CelebA/WildAligned_DatasetMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/CelebA/test_LR.json",
            "testfile": "/media/Workspace/Datasets/CelebA/testLR.txt"

        }
    },
    "MAAD": {
        "HR_HR": {
            "npy": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/croppedMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/test_HR.json",
            "testfile": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/testHR.txt"

        },
        "HR_LR": {
            "npy": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/WildcroppedMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/test_LR.json",
            "testfile": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/testLR.txt"

        },
        "LR_HR": {
            "npy": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/croppedMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/test_HR.json",
            "testfile": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/testHR.txt"

        },
        "LR_LR": {
            "npy": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/WildcroppedMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/test_LR.json",
            "testfile": "/media/Workspace/Datasets/VGG-Face2/data/vggface2_test/testLR.txt"

        }
    },
    "BUPT": {
        "HR_HR": {
            "npy": "/media/Workspace/Datasets/BUPT/DatasetHRMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/BUPT/test_HR.json",
            "testfile": "/media/Workspace/Datasets/BUPT/testHR.txt"

        },
        "HR_LR": {
            "npy": "/media/Workspace/Datasets/BUPT/DatasetLRMobileFaceNetNPCFaceOnHR.pt.npy",
            "json": "/media/Workspace/Datasets/BUPT/test_LR.json",
            "testfile": "/media/Workspace/Datasets/BUPT/testLR.txt"

        },
        "LR_HR": {
            "npy": "/media/Workspace/Datasets/BUPT/DatasetHRMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/BUPT/test_HR.json",
            "testfile": "/media/Workspace/Datasets/BUPT/testHR.txt"

        },
        "LR_LR": {
            "npy": "/media/Workspace/Datasets/BUPT/DatasetLRMobileFaceNetNPCFaceOnLR.pt.npy",
            "json": "/media/Workspace/Datasets/BUPT/test_LR.json",
            "testfile": "/media/Workspace/Datasets/BUPT/testLR.txt"

        }
    }

}
