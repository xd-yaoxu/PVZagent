"""Pure geometry; all pixel rectangles are half-open and rows/slots are 1-based."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_dict(cls, value):
        rect = cls(*(float(value[k]) for k in ('x', 'y', 'width', 'height')))
        if not all(math.isfinite(v) for v in (rect.x, rect.y, rect.width, rect.height)) or min(rect.width, rect.height) <= 0:
            raise ValueError('Invalid rectangle')
        return rect

    def contains(self, x, y):
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def as_dict(self):
        return dict(x=self.x, y=self.y, width=self.width, height=self.height)


class Calibration:
    def __init__(self, spec):
        self.spec = spec
        self.width, self.height = spec['canonical_size']
        self.grid = Rect.from_dict(spec['grid']['rect'])
        self.rows, self.columns = spec['grid']['rows'], spec['grid']['columns']
        self.slots = [Rect.from_dict(s['rect']) for s in spec['card_slots']]
        for rect in [self.grid, *self.slots, *[Rect.from_dict(v) for v in spec['regions'].values()]]:
            if rect.x < 0 or rect.y < 0 or rect.x + rect.width > self.width or rect.y + rect.height > self.height:
                raise ValueError('Calibration region outside canonical game image')

    def cell_center(self, row, column):
        if not isinstance(row, int) or not isinstance(column, int) or not 1 <= row <= self.rows or not 1 <= column <= self.columns:
            raise ValueError('Cell out of range')
        return (self.grid.x + (column - .5) * self.grid.width / self.columns,
                self.grid.y + (row - .5) * self.grid.height / self.rows)

    def cell_at(self, x, y):
        if not self.grid.contains(x, y):
            return None
        column = int((x - self.grid.x) / (self.grid.width / self.columns)) + 1
        row = int((y - self.grid.y) / (self.grid.height / self.rows)) + 1
        return row, column

    def slot_center(self, slot):
        if not isinstance(slot, int) or not 1 <= slot <= len(self.slots):
            raise ValueError('Slot out of range')
        rect = self.slots[slot - 1]
        return rect.x + rect.width / 2, rect.y + rect.height / 2

    def mirror_cell(self, row, column):
        self.cell_center(row, column)
        return self.rows + 1 - row, column


class CoordinateMap:
    """Use geometry from the same fresh capture; never cached desktop positions."""
    def __init__(self, calibration, metadata):
        self.calibration = calibration
        if list(metadata['image_pixels']) != [calibration.width, calibration.height]:
            raise ValueError('Capture size changed; recalibrate before generating targets')
        if not metadata['geometry_stable_during_capture']:
            raise ValueError('Window geometry changed during capture')
        if not metadata['display_mapping']['validated']:
            raise ValueError('Display mapping has not passed pixel alignment verification')
        self.viewport = Rect.from_dict(metadata['display_mapping']['game_viewport_screen'])
        self.state_id = metadata['state_id']

    def to_screen(self, x, y):
        if not Rect(0, 0, self.calibration.width, self.calibration.height).contains(x, y):
            raise ValueError('Point outside game image')
        return (self.viewport.x + x * self.viewport.width / self.calibration.width,
                self.viewport.y + y * self.viewport.height / self.calibration.height)

    def from_screen(self, x, y):
        if not self.viewport.contains(x, y):
            return None
        return ((x - self.viewport.x) * self.calibration.width / self.viewport.width,
                (y - self.viewport.y) * self.calibration.height / self.viewport.height)

    def point_record(self, kind, index, point):
        screen = self.to_screen(*point)
        return {'kind': kind, 'index': index, 'state_id': self.state_id,
                'image': list(point), 'screen_physical': list(screen),
                'screen_physical_rounded': [round(v) for v in screen]}

    def all_targets(self):
        c = self.calibration
        return [self.point_record('cell', [r, col], c.cell_center(r, col))
                for r in range(1, c.rows + 1) for col in range(1, c.columns + 1)] + [
                    self.point_record('card_slot', [s], c.slot_center(s)) for s in range(1, len(c.slots) + 1)]
