# read_dicom_wsi.py
import numpy as np
import pydicom
from pydicom.encaps import decode_data_sequence
from PIL import Image
import io

class TileWSI:
    def __init__(self, path):
        ds = pydicom.dcmread(path)
        self.ds = ds

        # Tile size (rows, columns)
        self.tile_h = ds.Rows
        self.tile_w = ds.Columns

        # Full-resolution size (width, height)
        self.full_w = ds[0x0048,0x0006].value
        self.full_h = ds[0x0048,0x0007].value

        # Number of tiles in grid
        self.tiles_x= int(np.ceil(self.full_w / self.tile_w))
        self.tiles_y= int(np.ceil(self.full_h / self.tile_h))

        # Decode JPEG tiles --> thats how the data is stored in DICOM
        self.tiles = decode_data_sequence(ds.PixelData)

    def tile_index(self, tx, ty):
        return ty * self.tiles_x + tx

    def get_tile(self, tx, ty):
        idx = self.tile_index(tx, ty)
        if idx >= len(self.tiles):
            return None
        tile_bytes = self.tiles[idx]
        return np.array(Image.open(io.BytesIO(tile_bytes)))

    def read_region(self, location, size):
        x, y= location
        w, h= size

        # tile grid
        tile_x_start= x // self.tile_w
        tile_y_start= y // self.tile_h
        tile_x_end = (x + w) // self.tile_w
        tile_y_end = (y + h) // self.tile_h

        region = np.zeros(((tile_y_end-tile_y_start+1)*self.tile_h,
                           (tile_x_end-tile_x_start+1)*self.tile_w,
                           3), dtype=np.uint8)

        for ty in range(tile_y_start, tile_y_end+1):
            for tx in range(tile_x_start, tile_x_end+1):
                tile = self.get_tile(tx, ty)
                if tile is None:
                    continue
                yy= (ty - tile_y_start)*self.tile_h
                xx= (tx - tile_x_start)*self.tile_w
                region[yy:yy+self.tile_h, xx:xx+self.tile_w]= tile

        off_x= x - tile_x_start* self.tile_w
        off_y= y - tile_y_start* self.tile_h
        return region[off_y:off_y+h, off_x:off_x+w]
