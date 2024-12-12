from preprocessing import preprocessingPipeline as pipe 
from fileHandler import loadFiles

def main():
    imageDataSet = loadFiles("photoDataset")
    pipe() 

if __name__ == "__main__":
    main()
