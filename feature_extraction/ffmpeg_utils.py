import ffmpeg
import numpy as np


def extract_frames(video_path, fps, size=None, crop=None, start=None, duration=None):
    print("input: ", input)
    try:
        
        if start is not None:
            cmd = ffmpeg.input(video_path, ss=start, t=duration)
        else:
            cmd = ffmpeg.input(video_path)
        
        if size is None:
            info = [s for s in ffmpeg.probe(video_path)["streams"] if s["codec_type"] == "video"][0]
            size = (info["width"], info["height"])
        elif isinstance(size, int):
            size = (size, size)

        if fps is not None:
            cmd = cmd.filter('fps', fps=fps)
        cmd = cmd.filter('scale', size[0], size[1])

        if crop is not None:
            cmd = cmd.filter('crop', f'in_w-{crop[0]}', f'in_h-{crop[1]}')
            size = (size[0] - crop[0], size[1] - crop[1])
        #had error here by not including .mp4 in the file name. 
        out, _ = (
            cmd.output("pipe:", format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, size[1], size[0], 3])
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    print("done returning video")
    return video