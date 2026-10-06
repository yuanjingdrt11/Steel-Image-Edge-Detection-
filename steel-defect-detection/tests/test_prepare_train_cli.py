"""Synthetic tests only: no production data, GPU, downloads, or real training."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image
import yaml

SRC = Path(__file__).resolve().parents[1] / 'src'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('prepare_mixed')
train = load('train_mixed')


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.out = self.root/'source', self.root/'out'
        self.source.mkdir()

    def sample(self, stem, size=(24, 16), boxes=None):
        if boxes is None:
            boxes = [('mamianmakeng', (6, 2, 18, 14)), ('jieba', (17, 1, 23, 7))]
        Image.new('RGB', size, (20, 40, 60)).save(self.source/f'{stem}.jpg')
        xml = f'<annotation><size><width>{size[0]}</width><height>{size[1]}</height></size>'
        for name, coords in boxes:
            xml += f'<object><name>{name}</name><bndbox>'
            xml += ''.join(f'<{k}>{v}</{k}>' for k, v in zip(('xmin', 'ymin', 'xmax', 'ymax'), coords))
            xml += '</bndbox></object>'
        (self.source/f'{stem}.xml').write_text(xml+'</annotation>')

    def args(self, *extra):
        return ['--src', str(self.source), '--out', str(self.out), '--tile', '16',
                '--val', '0', '--min-free-gb', '0', *extra]

    def run_prepare(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare.main(self.args(*extra))

    def test_raw_and_c_product_group_keys(self):
        for stem in ('123_Raw00_f_1', '123-Raw01-f_2'):
            self.assertEqual(prepare.group_key(stem), '123')
        self.assertEqual(prepare.group_key('C123_V03_F0002_uuid'), 'C123')
        self.assertEqual(prepare.group_key('other_name'), 'other_name')

    def test_group_split_full_validation_reproducible_and_source_unchanged(self):
        for stem in ('123_Raw00_f_1', '123-Raw01-f_2', 'C123_V00_F1', 'C123_V01_F2',
                     '456_Raw00_f_1', 'C456_V00_F1'):
            self.sample(stem)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}
        result = subprocess.run([sys.executable, str(SRC/'prepare_mixed.py'),
                                 *self.args('--val', '.5', '--seed', '3')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.out/'manifest.json').read_text())
        self.assertFalse(set(manifest['train_groups']) & set(manifest['val_groups']))
        self.assertEqual(len(manifest['val_groups']), 2)
        for record in manifest['records']:
            split, stem = record['split'], record['stem']
            self.assertTrue((self.out/'images'/split/f'{stem}.jpg').exists())
            self.assertTrue((self.out/'labels'/split/f'{stem}.txt').exists())
        self.assertFalse(list((self.out/'images/val').glob('*__tile*')))
        self.assertTrue(list((self.out/'images/train').glob('*__tile*')))
        self.assertEqual(hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()})
        data = yaml.safe_load((self.out/'data.yaml').read_text())
        self.assertEqual(data['names'][0], 'mamianmakeng')
        self.assertEqual(data['path'], str(self.out))
        self.out = self.root/'out2'
        again = self.run_prepare('--val', '.5', '--seed', '3', '--dry-run')
        self.assertEqual(again['val_groups'], manifest['val_groups'])
        self.assertFalse(self.out.exists())

    def test_tile_intersections_clip_and_normalize_all_targets(self):
        self.sample('a_Raw0', boxes=[('mamianmakeng', (0, 2, 12, 14)),
                                    ('jieba', (17, 1, 23, 7))])
        self.run_prepare()
        first = (self.out/'labels/train/a_Raw0__tile_0_0.txt').read_text().splitlines()
        self.assertEqual(len(first), 1)  # The second object is entirely outside this tile.
        self.assertEqual([float(x) for x in first[0].split()], [0, .375, .5, .75, .75])
        second = (self.out/'labels/train/a_Raw0__tile_8_0.txt').read_text().splitlines()
        self.assertEqual(len(second), 2)  # Include a truncated neighboring object as well.
        self.assertEqual([float(x) for x in second[0].split()], [0, .125, .5, .25, .75])
        self.assertEqual([float(x) for x in second[1].split()], [1, .75, .25, .375, .375])
        with Image.open(self.out/'images/train/a_Raw0__tile_8_0.jpg') as image:
            self.assertEqual(image.size, (16, 16))

    def test_default_1280_small_image_empty_targets_and_border_boxes(self):
        self.assertEqual(prepare.parser().parse_args([]).tile, 1280)
        self.sample('small', size=(8, 4), boxes=[('jieba', (-2, -3, 20, 12))])
        self.sample('empty', size=(8, 4), boxes=[])
        self.run_prepare('--tile', '1280')
        self.assertEqual((self.out/'labels/train/small__tile_0_0.txt').read_text(),
                         '1 0.50000000 0.50000000 1.00000000 1.00000000\n')
        self.assertEqual((self.out/'labels/train/empty__tile_0_0.txt').read_text(), '')

    def test_disk_preflight_and_override(self):
        self.sample('a')
        usage = types.SimpleNamespace(free=prepare.GIB)
        with mock.patch.object(prepare.shutil, 'disk_usage', return_value=usage):
            with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                self.run_prepare('--min-free-gb', '35')
            self.assertIn('required 35.00 GiB', err.getvalue())
            self.assertFalse(self.out.exists())
            self.run_prepare('--min-free-gb', '0')
        self.assertEqual(prepare.parser().parse_args([]).min_free_gb, 35)

    def test_output_estimate_still_required_after_override(self):
        self.sample('a')
        with mock.patch.object(prepare.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=1)):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_unpaired_rejected_and_explicit_opt_in_manifest(self):
        self.sample('paired')
        Image.new('RGB', (4, 4)).save(self.source/'missing.jpg')
        (self.source/'orphan.xml').write_text('<annotation/>')
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            self.run_prepare()
        self.assertIn('--allow-unpaired', err.getvalue())
        manifest = json.loads((self.out/'manifest.json').read_text())
        self.assertEqual(len(manifest['unpaired_images']), 1)
        self.assertEqual(len(manifest['unpaired_annotations']), 1)
        self.assertFalse((self.out/'images').exists())
        manifest = self.run_prepare('--allow-unpaired')
        self.assertEqual(manifest['images'], 1)
        self.assertEqual(len(manifest['unpaired_images']), 1)
        self.assertFalse((self.out/'images/train/missing.jpg').exists())

    def test_link_fallbacks_and_no_overwrite(self):
        src = self.source/'x.jpg'
        src.write_bytes(b'original')
        dst = self.root/'linked.jpg'
        self.assertEqual(prepare.link_image(src, dst), 'hardlink')
        self.assertEqual(src.stat().st_ino, dst.stat().st_ino)
        with self.assertRaises(FileExistsError):
            prepare.link_image(src, dst)
        with mock.patch.object(prepare.os, 'link', side_effect=OSError('cross-device')):
            self.assertEqual(prepare.link_image(src, self.root/'sym.jpg'), 'symlink')
            with mock.patch.object(Path, 'symlink_to', side_effect=OSError('unsupported')):
                copy = self.root/'copy.jpg'
                self.assertEqual(prepare.link_image(src, copy), 'copy')
                self.assertEqual(copy.read_bytes(), b'original')

    def test_nonempty_or_nested_output_refused(self):
        self.sample('a')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_prepare('--out', str(self.source/'out'))
        self.out.mkdir()
        sentinel = self.out/'keep'
        sentinel.write_text('keep')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_prepare()
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_tile_limit_dedup_and_seed_reproducibility(self):
        boxes = [(0, i*10, 0, i*10+8, 8) for i in range(15)]
        import random
        tiles = prepare.select_tiles(boxes, 160, 16, 16, 4, random.Random(5))
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles, prepare.select_tiles(boxes, 160, 16, 16, 4, random.Random(5)))
        self.assertEqual(len(prepare.select_tiles(boxes*2, 160, 16, 16, 100, random.Random(5))), 15)

    def test_cli_help(self):
        for script in ('prepare_mixed.py', 'train_mixed.py'):
            result = subprocess.run([sys.executable, str(SRC/script), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)


class TrainTests(unittest.TestCase):
    def setUp(self):
        self.yolo = mock.Mock()
        self.yolo.return_value.names = {i: 'person' if i == 0 else str(i) for i in range(80)}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'synthetic.yaml'
        self.data.write_text('names: []\n')
        patch = mock.patch.dict(sys.modules, {'ultralytics': types.SimpleNamespace(YOLO=self.yolo)})
        patch.start()
        self.addCleanup(patch.stop)

    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return train.main(['--data', str(self.data), '--project', str(self.root/'runs'), *args])

    def test_defaults_coco_and_no_one_epoch_stop(self):
        self.invoke([])
        self.assertEqual(Path(self.yolo.call_args.args[0]).name, 'yolo11s.pt')
        kw = self.yolo.return_value.train.call_args.kwargs
        for key, value in {'name': 'mixed_scale', 'epochs': 150, 'imgsz': 1024, 'batch': 4,
                           'workers': 4, 'device': '0', 'patience': 40, 'amp': True,
                           'mixup': 0, 'resume': False}.items():
            self.assertEqual(kw[key], value)
        self.assertLessEqual(kw['mosaic'], .2)
        self.assertEqual([kw[k] for k in ('hsv_h', 'hsv_s', 'hsv_v')], [0, 0, 0])
        self.yolo.return_value.add_callback.assert_not_called()
        self.assertNotIn('time', kw)

    def test_cli_overrides(self):
        self.invoke(['--data', str(self.data), '--project', 'test-runs', '--name', 'test',
                     '--epochs', '8', '--imgsz', '640', '--batch', '2', '--workers', '0', '--device', 'cpu'])
        kw = self.yolo.return_value.train.call_args.kwargs
        for key, value in {'data': str(self.data), 'project': 'test-runs', 'name': 'test',
                           'epochs': 8, 'imgsz': 640, 'batch': 2, 'workers': 0, 'device': 'cpu'}.items():
            self.assertEqual(kw[key], value)

    def test_resume_only_last_and_preserves_saved_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            last = Path(tmp)/'last.pt'
            last.touch()
            self.invoke(['--resume', str(last), '--device', 'cpu'])
            self.yolo.assert_called_once_with(str(last))
            self.yolo.return_value.train.assert_called_once_with(resume=True, device='cpu', workers=4)
            self.yolo.reset_mock()
            best = Path(tmp)/'best.pt'
            best.touch()
            for checkpoint in (best, Path(tmp)/'missing'/'last.pt'):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.invoke(['--resume', str(checkpoint)])
            self.yolo.assert_not_called()

    def test_missing_data_nonempty_run_and_non_coco_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(['--data', str(self.root/'missing.yaml')])
        run = self.root/'runs'/'mixed_scale'
        run.mkdir(parents=True)
        (run/'keep').write_text('keep')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke([])
        self.yolo.assert_not_called()
        self.yolo.return_value.names = {0: 'mamianmakeng'}
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(['--name', 'fresh'])
        self.yolo.return_value.train.assert_not_called()

    def test_root_weights_and_download_name_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(train, 'PROJECT_ROOT', Path(tmp)):
            self.invoke([])
            self.yolo.assert_called_with('yolo11s.pt')
            weights = Path(tmp)/'yolo11s.pt'
            weights.touch()
            self.invoke([])
            self.yolo.assert_called_with(str(weights))


if __name__ == '__main__':
    unittest.main()
