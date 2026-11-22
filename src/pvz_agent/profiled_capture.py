"""Read-only capture with lossless fast encoding and explicit stage timings."""
import ctypes.wintypes as w
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import uuid
from PIL import Image,ImageChops,ImageGrab,ImageStat
from .windows_capture import WindowsCapture


def encode_png(image,level=1):
    if type(level) is not int or not 0<=level<=9:raise ValueError('PNG compression level0..9 required')
    buffer=BytesIO();image.save(buffer,format='PNG',compress_level=level);return buffer.getvalue()


class ProfiledWindowsCapture(WindowsCapture):
    def capture(self,output_dir,label='profiled',expected_size=(800,600)):
        if not label.replace('-','').isalnum():raise ValueError('Invalid capture label')
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        started=datetime.now(timezone.utc).isoformat();tick=time.monotonic_ns();stages={};mark=time.perf_counter()
        def stage(name):
            nonlocal mark
            now=time.perf_counter();stages[name]=(now-mark)*1000;mark=now
        self.u.GetForegroundWindow.argtypes=[];self.u.GetForegroundWindow.restype=w.HWND
        foreground_before=int(self.u.GetForegroundWindow() or 0)
        before=self.inspect();stage('inspect_before_ms')
        image=ImageGrab.grab(window=before['game']['hwnd']).convert('RGB');stage('native_grab_ms')
        if image.size!=tuple(expected_size):raise RuntimeError(f'Unexpected game capture {image.size}; recalibration required')
        if max(ImageStat.Stat(image).var)<10:raise RuntimeError('Black/blank capture')
        mapping=self._mapping(before);target=mapping['game_viewport_screen']
        bbox=(round(target['x']),round(target['y']),round(target['x']+target['width']),round(target['y']+target['height']))
        with self.physical_coordinates():display=ImageGrab.grab(bbox=bbox,all_screens=True).convert('RGB')
        stage('display_grab_and_mapping_ms')
        after=self.inspect();foreground_after=int(self.u.GetForegroundWindow() or 0);stage('inspect_after_ms')
        stable=before==after
        normalized=display.resize(image.size,Image.Resampling.LANCZOS)
        error=sum(ImageStat.Stat(ImageChops.difference(image,normalized)).mean)/3
        mapping.update(alignment_mean_absolute_error=error,validated=bool(stable and error<=18),validation_threshold=18)
        stage('alignment_and_validation_ms')
        # Timestamp describes captured pixels before encoding/persistence.
        stamp=datetime.now(timezone.utc).isoformat();state_id=f'{label}-{stamp.replace(":","-").replace(".","-")}-{uuid.uuid4().hex[:6]}'
        native=encode_png(image);stage('native_encode_ms')
        visible=encode_png(display);stage('display_encode_ms')
        native_file=output_dir/(state_id+'.png');display_file=output_dir/(state_id+'-display.png')
        native_file.write_bytes(native);display_file.write_bytes(visible);stage('image_write_ms')
        native_sha=hashlib.sha256(native).hexdigest();display_sha=hashlib.sha256(visible).hexdigest()
        pixel_sha=hashlib.sha256(image.tobytes()).hexdigest();stage('hash_ms')
        allowed=[before['game']['hwnd']]+([before['scaler']['hwnd']] if before.get('scaler') else [])
        meta=dict(schema_version=2,state_id=state_id,capture_started_at=started,capture_time=stamp,monotonic_ns=tick,
            image_pixels=list(image.size),image_file=native_file.name,display_file=display_file.name,
            source='Pillow_single_window_image_capture',windows=before,geometry_stable_during_capture=stable,
            display_mapping=mapping,image_sha256=native_sha,display_sha256=display_sha,
            image_pixel_sha256=pixel_sha,pixel_hash_semantics='SHA256 of canonical RGB bytes; dimensions in image_pixels',
            png_compress_level=1,profile=stages,capture_duration_ms=(time.monotonic_ns()-tick)/1e6,
            capture_duration_scope='Through image persistence and hashes, before metadata persistence.',
            foreground_before=foreground_before,foreground_after=foreground_after,
            foreground_stable=foreground_before==foreground_after,
            foreground_game_or_scaler=foreground_before==foreground_after and foreground_before in allowed,
            limitations=['No input or activation. Alignment is not complete occlusion/scene detection.',
                'File SHA changes with encoding; RGB pixel SHA does not. Revalidate before any future action.'])
        metadata_file=output_dir/(state_id+'.json')
        metadata_file.write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return native_file,metadata_file,meta
