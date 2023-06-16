import timm
def doFFMPeg(videoLink):
    """
    Does the FFMPEG operations on the video so that we can get it into a good format. 
    """
    #need image dims to b 3 x 224, 224, loaded into range 0,1 normalized with mean .485, .456, .406, and std .229, .224, .225
    model = timm.create_model('cspresnext50', pretrained = True)
