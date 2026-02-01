import json
import os
import re
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset
from ..data_utils import pil_img2rgb
from ..distributed_iterable_dataset import DistributedIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class ThinkRefineEditIterableDataset(InterleavedBaseIterableDataset, DistributedIterableDataset):
    def __init__(
        self, 
        dataset_name, 
        transform, 
        tokenizer, 
        vit_transform,
        jsonl_path_list, 
        data_dir_list, 
        num_used_data,
        local_rank=0, 
        world_size=1, 
        num_workers=8, 
        data_status=None,
        shuffle_lines=True, 
        shuffle_seed=0,
        image_prefix_dir=None,
    ):
        DistributedIterableDataset.__init__(self, dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform  
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.image_prefix_dir = image_prefix_dir or ""
        
        self.start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
        
        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            num_used_data, 
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

    # -------------------- 新增：统一读取 JSON 数组 / JSONL --------------------
    def _read_records_from_path(self, path):
        """
        返回一个 list[str]，每个元素是一条 JSON 记录的字符串。
        - 如果文件是 JSON 数组：[ {...}, {...}, ... ]，则将每个元素 json.dumps 成一条字符串。
        - 如果文件是 NDJSON/JSONL，则逐行 strip 后返回非空行。
        """
        # 先探测首个非空字符
        with open(path, 'r', encoding='utf-8') as f:
            head = f.read(1024)
        first_sig = None
        for ch in head:
            if not ch.isspace():
                first_sig = ch
                break

        # JSON 数组
        if first_sig == '[':
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    arr = json.load(f)
                except Exception:
                    # 若整体 load 失败，抛出给上层，让你看到更早的错误
                    raise
            if not isinstance(arr, list):
                raise ValueError(f"File looks like a JSON array but parsed type={type(arr)}: {path}")
            # 统一转为“每条记录一行 JSON 字符串”，便于 parse_row 复用
            return [json.dumps(obj, ensure_ascii=False) for obj in arr]

        # 否则按 JSONL 逐行读取
        with open(path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines()]
        # 过滤空行
        return [ln for ln in lines if ln]

    def get_data_paths(self, jsonl_path_list, num_used_data, shuffle_lines, shuffle_seed):
        data_paths = []
        if not isinstance(num_used_data, list):
            num_used_data = [num_used_data] * len(jsonl_path_list)

        for jsonl_path, num_data_point in zip(jsonl_path_list, num_used_data):
            # 统一读取为“记录列表（每条是 JSON 字符串）”
            raw_data = self._read_records_from_path(jsonl_path)

            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            
            # Convert 'None' string to None type
            if num_data_point == 'None':
                num_data_point = None

            if num_data_point is not None and int(num_data_point) > 0:
                raw_data = raw_data[:int(num_data_point)]

            data_paths.extend(raw_data)
        return data_paths
    # -------------------- 新增方法到此为止 --------------------

    def extract_image_references(self, text):
        pattern = r'<image_start>\[([^\]]+)\]<image_end>'
        matches = re.findall(pattern, text)
        return matches

    def replace_image_references(self, text):
        pattern = r'<image_start>\[([^\]]+)\]<image_end>'
        return re.sub(pattern, '<IMAGE_PLACEHOLDER>', text)

    def remove_thought_patterns(self, text):
        pattern = r'THOUGHT\s*\d+:\s*'
        return re.sub(pattern, '', text)

    def load_image_safely(self, data_item, image_key):
        if image_key not in data_item or data_item[image_key] is None:
            return None
        image_path = data_item[image_key]
        if isinstance(image_path, list):
            image_path = image_path[0]
        if self.image_prefix_dir is not None:
            full_path = os.path.join(self.image_prefix_dir, image_path)
        else:
            full_path = image_path
        try:
            return pil_img2rgb(Image.open(full_path))
        except Exception as e:
            print(f"Failed to load image {full_path}: {e}")
            return None

    def parse_row(self, json_line):
        try:
            data_item = json.loads(json_line.strip())
        except:
            traceback.print_exc()
            return {}
       

        # prompt = "You are an intelligent assistant specialized in reasoning-based response generation, self-evaluation, and iterative refinement through step-by-step interleaved text-visual chain of thought. For each user question, follow this three-step process: (1) Initial Output Generation: you should first think about the planning process using your current knowledge in the mind and then generate the initial image output. The planning process is enclosed within <think>...</think>. (2)Self-Evaluation: Critically assess your generated output (e.g.,  consistency with the question, attributes correctness, style consistency, realism, visual quality of the image). (3)Refinement: if issues are found, write a brief modification suggestion enclosed in <sugg>...</sugg> and regenerate an improved image output accordingly.'"
        prompt = "You are an intelligent assistant specialized in image generation and editing, self-evaluation, and iterative refinement through step-by-step interleaved text-visual chain of thought. For each user instruction, follow this three-step process: (1) Initial Output Generation: generate the initial image follwing user instruction. (2)Self-Evaluation: Critically assess your generated output (e.g., consistency with the instruction, attributes correctness, style consistency, realism, visual quality of the image). (3)Refinement: if Image needs refine, write a brief modification suggestion enclosed in <sugg>...</sugg> and regenerate an improved image output accordingly.'"
        question = data_item.get('prompt', '')
        question = f'{question}'
        thinking = data_item.get('explanation', '')
        thinking = f"<think>{thinking.strip()}</think>"
        reasoning_trace = data_item.get('Text Reasoning Trace', '')
        reasoning_trace = f'{reasoning_trace}'

        if not question or not reasoning_trace:
            return {}
        
        before_image = self.load_image_safely(data_item, 'before_image_path')
        after_image = self.load_image_safely(data_item, 'after_image_path')

        data = self._init_data()
        data = self._add_text(data, prompt, need_loss=False, enable_cfg=True)
        data = self._add_text(data, question, need_loss=False, enable_cfg=True)
        data = self._add_image(
                data,
                before_image, 
                need_loss=False,
                need_vae=True,
                need_vit=True,
        )

        if thinking != "":
            data = self._add_text(data, thinking, need_loss=False, enable_cfg=True)

        
        data = self._add_image(
                data,
                after_image, 
                need_loss=False,
                need_vae=True,
                need_vit=True,
        )
 


        
        image_refs = self.extract_image_references(reasoning_trace)
        loaded_images = []
        for image_ref in image_refs:
            image = self.load_image_safely(data_item, image_ref)
            if image is not None:
                loaded_images.append(image)
            else:
                print(f"Skipping sample due to missing image: {image_ref}")
                return {}

        clean_reasoning_trace = self.replace_image_references(reasoning_trace)
        clean_reasoning_trace = self.remove_thought_patterns(clean_reasoning_trace)
        text_parts = clean_reasoning_trace.split('<IMAGE_PLACEHOLDER>')
        if len(text_parts) != len(loaded_images) + 1:
            print(f"Mismatch between text parts ({len(text_parts)}) and images ({len(loaded_images)})")
            return {}

        # print(text_parts)

        for i, text_part in enumerate(text_parts):
         
            next_token_label = self.start_of_image
            wrapped_text = f"<sugg>{text_part.strip()}</sugg>"
            data = self._add_text(data, wrapped_text, need_loss=True, enable_cfg=True, next_token_label=next_token_label)
            if i < len(loaded_images):
                data = self._add_image(
                    data,
                    loaded_images[i], 
                    need_loss=True,
                    need_vae=True,
                    need_vit=True,
                    enable_cfg=True,
                )

      
        return data

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, json_line in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data = self.parse_row(json_line)
                    if len(data) == 0:
                        continue
                    has_loss = any(item['loss'] for item in data['sequence_plan'])
                    if not has_loss:
                        print('No loss defined, skipped.')
                        continue
                    data['data_indexes'] = {
                        "data_indexes": row_idx,
                        "worker_id": worker_id, 
                        "dataset_name": self.dataset_name,
                    }
                    yield data
                except Exception as e:
                    print(f"Error processing row {row_idx}: {e}")
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
