from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Union

import hashlib
import json
import logging
import random
from itertools import groupby
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from haystack.modeling.data_handler.dataloader import NamedDataLoader
from haystack.modeling.data_handler.processor import Processor, SquadProcessor
from haystack.utils.experiment_tracking import Tracker as tracker
from haystack.modeling.visual import TRACTOR_SMALL

if TYPE_CHECKING:
    from haystack.nodes import FARMReader

logger = logging.getLogger(__name__)


class DataSilo:
    """Generates and stores PyTorch DataLoader objects for the train, dev and test datasets.
    Relies upon functionality in the processor to do the conversion of the data. Will also
    calculate and display some statistics.
    """

    def __init__(
        self,
        processor: Processor,
        batch_size: int,
        eval_batch_size: Optional[int] = None,
        distributed: bool = False,
        automatic_loading: bool = True,
        max_multiprocessing_chunksize: int = 512,
        max_processes: int = 128,
        multiprocessing_strategy: Optional[str] = None,
        caching: bool = False,
        cache_path: Path = Path("cache/data_silo"),
        siamese: bool = False,
    ):
        """
        :param processor: A dataset specific Processor object which will turn input (file or dict) into a Pytorch Dataset.
        :param batch_size: The size of batch that should be returned by the DataLoader for the training set.
        :param eval_batch_size: The size of batch that should be returned by the DataLoaders for the dev and test set.
        :param distributed: Set to True if you are running in a distributed evn, e.g. using DistributedDataParallel.
                            The DataSilo will init the DataLoader with a DistributedSampler() to distribute batches.
        :param automatic_loading: Set to False, if you don't want to automatically load data at initialization.
        :param max_multiprocessing_chunksize: max possible value for chunksize as calculated by `calc_chunksize()`
            in `haystack.basics.utils`. For certain cases like lm_finetuning, a smaller value can be set, as the default chunksize
            values are rather large that might cause memory issues.
        :param max_processes: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing or make debugging easier.
                              .. deprecated:: 1.9
                                    Multiprocessing has been removed in 1.9. This parameter will be ignored.
        :multiprocessing_strategy: Set the multiprocessing sharing strategy, this can be one of file_descriptor/file_system depending on your OS.
                                   If your system has low limits for the number of open file descriptors, and you can’t raise them,
                                   you should use the file_system strategy.
                                   .. deprecated:: 1.9
                                        Multiprocessing has been removed in 1.9. This parameter will be ignored.
        :param caching: save the processed datasets on disk to save time/compute if the same train data is used to run
                        multiple experiments. Each cache has a checksum based on the train_filename of the Processor
                        and the batch size.
        :param cache_path: root dir for storing the datasets' cache.
        """
        self.distributed = distributed
        self.processor = processor
        self.data = {}  # type: Dict
        self.batch_size = batch_size
        self.class_weights = None
        self.max_processes = max_processes
        self.multiprocessing_strategy = multiprocessing_strategy
        self.max_multiprocessing_chunksize = max_multiprocessing_chunksize
        self.caching = caching
        self.cache_path = cache_path
        self.tensor_names = None
        self.siamese = siamese
        if eval_batch_size is None:
            self.eval_batch_size = batch_size
        else:
            self.eval_batch_size = eval_batch_size

        if len(self.processor.tasks) == 0:
            raise Exception(
                "No task initialized. Try initializing the processor with a metric and a label list. "
                "Alternatively you can add a task using Processor.add_task()"
            )

        loaded_from_cache = False
        if self.caching:  # Check if DataSets are present in cache
            checksum = self._get_checksum()
            dataset_path = self.cache_path / checksum

            if dataset_path.exists():
                self._load_dataset_from_cache(dataset_path)
                loaded_from_cache = True

        if not loaded_from_cache and automatic_loading:
            # In most cases we want to load all data automatically, but in some cases we rather want to do this
            # later or load from dicts instead of file
            self._load_data()

    def _get_dataset(self, filename: Optional[Union[str, Path]], dicts: Optional[List[Dict]] = None):
        if not filename and not dicts:
            raise ValueError("You must either supply `filename` or `dicts`")

        # loading dicts from file (default)
        if dicts is None:
            dicts = list(self.processor.file_to_dicts(filename))  # type: ignore
            # shuffle list of dicts here if we later want to have a random dev set splitted from train set
            if str(self.processor.train_filename) in str(filename):
                if not self.processor.dev_filename:
                    if self.processor.dev_split > 0.0:
                        random.shuffle(dicts)

        num_dicts = len(dicts)
        datasets = []
        problematic_ids_all = set()
        batch_size = self.max_multiprocessing_chunksize

        for i in tqdm(range(0, num_dicts, batch_size), desc="Preprocessing dataset", unit=" Dicts"):
            # dicts[i]: query, passage
            processing_batch = dicts[i : i + batch_size]
            dataset, tensor_names, problematic_sample_ids = self.processor.dataset_from_dicts(
                dicts=processing_batch, indices=list(range(len(processing_batch)))  # TODO remove indices
            )
            datasets.append(dataset)
            problematic_ids_all.update(problematic_sample_ids)

        self.processor.log_problematic(problematic_ids_all)
        datasets = [d for d in datasets if d]

        concat_datasets = ConcatDataset(datasets)  # type: Dataset
        return concat_datasets, tensor_names

    def _load_data(
        self,
        train_dicts: Optional[List[Dict]] = None,
        dev_dicts: Optional[List[Dict]] = None,
        test_dicts: Optional[List[Dict]] = None,
    ):
        """
        Loading the train, dev and test datasets either from files (default) or from supplied dicts.
        The processor is called to handle the full conversion from "raw data" to a Pytorch Dataset.
        The resulting datasets are loaded into DataSilo.data

        :param train_dicts: (Optional) dicts containing examples for training.
        :param dev_dicts: (Optional) dicts containing examples for dev.
        :param test_dicts: (Optional) dicts containing examples for test.
        :return: None
        """

        logger.info("\nLoading data into the data silo ... %s", TRACTOR_SMALL)
        # train data
        logger.info("LOADING TRAIN DATA")
        logger.info("==================")
        if train_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["train"], self.tensor_names = self._get_dataset(filename=None, dicts=train_dicts)
        elif self.processor.train_filename:
            # or from a file (default)
            train_file = self.processor.data_dir / self.processor.train_filename
            logger.info("Loading train set from: %s ", train_file)
            self.data["train"], self.tensor_names = self._get_dataset(train_file)
        else:
            logger.info("No train set is being loaded")
            self.data["train"] = None

        # dev data
        logger.info("")
        logger.info("LOADING DEV DATA")
        logger.info("=================")
        if dev_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["dev"], self.tensor_names = self._get_dataset(filename=None, dicts=dev_dicts)
        elif self.processor.dev_filename:
            # or from file (default)
            dev_file = self.processor.data_dir / self.processor.dev_filename
            logger.info("Loading dev set from: %s", dev_file)
            self.data["dev"], _ = self._get_dataset(dev_file)
        elif self.processor.dev_split > 0.0:
            # or split it apart from train set
            logger.info("Loading dev set as a slice of train set")
            self._create_dev_from_train()
        else:
            logger.info("No dev set is being loaded")
            self.data["dev"] = None

        logger.info("")
        logger.info("LOADING TEST DATA")
        logger.info("=================")
        # test data
        if test_dicts:
            # either from supplied dicts
            logger.info("Loading train set from supplied dicts ")
            self.data["test"], self.tensor_names = self._get_dataset(filename=None, dicts=test_dicts)
        elif self.processor.test_filename:
            # or from file (default)
            test_file = self.processor.data_dir / self.processor.test_filename
            logger.info("Loading test set from: %s", test_file)
            if self.tensor_names:
                self.data["test"], _ = self._get_dataset(test_file)
            else:
                self.data["test"], self.tensor_names = self._get_dataset(test_file)
        else:
            logger.info("No test set is being loaded")
            self.data["test"] = None

        if self.caching:
            self._save_dataset_to_cache()

        # derive stats and meta data
        self._calculate_statistics()
        # self.calculate_class_weights()

        self._initialize_data_loaders()

    def _load_dataset_from_cache(self, cache_dir: Path):
        """
        Load serialized dataset from a cache.
        """
        logger.info("Loading datasets from cache at %s", cache_dir)
        self.data["train"] = torch.load(cache_dir / "train_dataset")

        dev_dataset_path = cache_dir / "dev_dataset"
        if dev_dataset_path.exists():
            self.data["dev"] = torch.load(dev_dataset_path)
        else:
            self.data["dev"] = None

        test_dataset_path = cache_dir / "test_dataset"
        if test_dataset_path.exists():
            self.data["test"] = torch.load(test_dataset_path)
        else:
            self.data["test"] = None

        self.tensor_names = torch.load(cache_dir / "tensor_names")

        # derive stats and meta data
        self._calculate_statistics()
        # self.calculate_class_weights()

        self._initialize_data_loaders()

    def _get_checksum(self):
        """
        Get checksum based on a dict to ensure validity of cached DataSilo
        """
        # keys in the dict identifies uniqueness for a given DataSilo.
        payload_dict = {
            "train_filename": str(Path(self.processor.train_filename).absolute()),
            "data_dir": str(self.processor.data_dir.absolute()),
            "max_seq_len": self.processor.max_seq_len,
            "dev_split": self.processor.dev_split,
            "tasks": self.processor.tasks,
        }
        checksum = get_dict_checksum(payload_dict)
        return checksum

    def _save_dataset_to_cache(self):
        """
        Serialize and save dataset to a cache.
        """
        checksum = self._get_checksum()

        cache_dir = self.cache_path / checksum
        cache_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.data["train"], cache_dir / "train_dataset")

        if self.data["dev"]:
            torch.save(self.data["dev"], cache_dir / "dev_dataset")

        if self.data["test"]:
            torch.save(self.data["test"], cache_dir / "test_dataset")

        torch.save(self.tensor_names, cache_dir / "tensor_names")
        logger.info("Cached the datasets at %s", cache_dir)

    def _initialize_data_loaders(self):
        """
        Initializing train, dev and test data loaders for the already loaded datasets.
        """

        if self.data["train"] is not None:
            if self.distributed:
                sampler_train = DistributedSampler(self.data["train"])
            else:
                sampler_train = RandomSampler(self.data["train"])

            data_loader_train = NamedDataLoader(
                dataset=self.data["train"],
                sampler=sampler_train,
                batch_size=self.batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_train = None

        if self.data["dev"] is not None:
            data_loader_dev = NamedDataLoader(
                dataset=self.data["dev"],
                sampler=SequentialSampler(self.data["dev"]),
                batch_size=self.eval_batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_dev = None

        if self.data["test"] is not None:
            data_loader_test = NamedDataLoader(
                dataset=self.data["test"],
                sampler=SequentialSampler(self.data["test"]),
                batch_size=self.eval_batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_test = None

        self.loaders = {"train": data_loader_train, "dev": data_loader_dev, "test": data_loader_test}

    def _create_dev_from_train(self):
        """
        Split a dev set apart from the train dataset.
        """
        n_dev = int(self.processor.dev_split * len(self.data["train"]))
        n_train = len(self.data["train"]) - n_dev

        train_dataset, dev_dataset = self.random_split_ConcatDataset(self.data["train"], lengths=[n_train, n_dev])
        self.data["train"] = train_dataset
        if len(dev_dataset) > 0:
            self.data["dev"] = dev_dataset
        else:
            logger.warning("No dev set created. Please adjust the dev_split parameter.")

        logger.info(
            "Took %s samples out of train set to create dev set (dev split is roughly %s)",
            len(dev_dataset),
            self.processor.dev_split,
        )

    def random_split_ConcatDataset(self, ds: ConcatDataset, lengths: List[int]):
        """
        Roughly split a Concatdataset into non-overlapping new datasets of given lengths.
        Samples inside Concatdataset should already be shuffled.

        :param ds: Dataset to be split.
        :param lengths: Lengths of splits to be produced.
        """
        if sum(lengths) != len(ds):
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        try:
            idx_dataset = np.where(np.array(ds.cumulative_sizes) > lengths[0])[0][0]
        except IndexError:
            raise Exception(
                "All dataset chunks are being assigned to train set leaving no samples for dev set. "
                "Either consider increasing dev_split or setting it to 0.0\n"
                f"Cumulative chunk sizes: {ds.cumulative_sizes}\n"
                f"train/dev split: {lengths}"
            )

        assert idx_dataset >= 1, (
            "Dev_split ratio is too large, there is no data in train set. "
            f"Please lower dev_split = {self.processor.dev_split}"
        )

        train = ConcatDataset(ds.datasets[:idx_dataset])  # type: Dataset
        test = ConcatDataset(ds.datasets[idx_dataset:])  # type: Dataset
        return train, test

    def _calculate_statistics(self):
        """Calculate and log simple summary statistics of the datasets"""
        logger.info("")
        logger.info("DATASETS SUMMARY")
        logger.info("================")

        self.counts = {}
        clipped = -1
        ave_len = -1

        if self.data["train"]:
            self.counts["train"] = len(self.data["train"])
            if "input_ids" in self.tensor_names and not self.siamese:
                clipped, ave_len, seq_lens, max_seq_len = self._calc_length_stats_single_encoder()
            elif "input_ids" in self.tensor_names and self.siamese:
                clipped, ave_len, seq_lens, max_seq_len = self._calc_length_stats_siamese_encoder()
            elif "query_input_ids" in self.tensor_names and "passage_input_ids" in self.tensor_names:
                clipped, ave_len, seq_lens, max_seq_len = self._calc_length_stats_biencoder()
            else:
                logger.warning(
                    "Could not compute length statistics because 'input_ids' or 'query_input_ids' and 'passage_input_ids' are missing."
                )
                clipped = -1
                ave_len = -1
        else:
            self.counts["train"] = 0

        if self.data["dev"]:
            self.counts["dev"] = len(self.data["dev"])
        else:
            self.counts["dev"] = 0

        if self.data["test"]:
            self.counts["test"] = len(self.data["test"])
        else:
            self.counts["test"] = 0

        logger.info("Examples in train: %s", self.counts["train"])
        logger.info("Examples in dev  : %s", self.counts["dev"])
        logger.info("Examples in test : %s", self.counts["test"])
        logger.info("Total examples   : %s", self.counts["train"] + self.counts["dev"] + self.counts["test"])
        logger.info("")
        if self.data["train"]:
            # SquadProcessor does not clip sequences, but splits them into multiple samples
            if "input_ids" in self.tensor_names and not isinstance(self.processor, SquadProcessor):
                logger.info("Longest sequence length observed after clipping:     %s", max(seq_lens))
                logger.info("Average sequence length after clipping: %s", ave_len)
                logger.info("Proportion clipped:      %s", clipped)
                if clipped > 0.5:
                    logger.info(
                        "[Haystack Tip] %s%% of your samples got cut down to %s tokens. "
                        "Consider increasing max_seq_len "
                        "(the maximum value allowed with the current model is max_seq_len=%s, "
                        "if this is not enough consider splitting the document in smaller units or changing the model). "
                        "This will lead to higher memory consumption but is likely to improve your model performance",
                        round(clipped * 100, 1),
                        max_seq_len,
                        self.processor.tokenizer.model_max_length,
                    )
            elif "query_input_ids" in self.tensor_names and "passage_input_ids" in self.tensor_names:
                logger.info(
                    "Longest query length observed after clipping: %s   - for max_query_len: %s",
                    max(seq_lens[0]),
                    max_seq_len[0],
                )
                logger.info("Average query length after clipping:          %s", ave_len[0])
                logger.info("Proportion queries clipped:                   %s", clipped[0])
                logger.info("")
                logger.info(
                    "Longest passage length observed after clipping: %s   - for max_passage_len: %s",
                    max(seq_lens[1]),
                    max_seq_len[1],
                )
                logger.info("Average passage length after clipping:          %s", ave_len[1])
                logger.info("Proportion passages clipped:                    %s", clipped[1])

        tracker.track_params(
            {
                "n_samples_train": self.counts["train"],
                "n_samples_dev": self.counts["dev"],
                "n_samples_test": self.counts["test"],
                "batch_size": self.batch_size,
                "ave_seq_len": ave_len,
                "clipped": clipped,
            }
        )

    def _calc_length_stats_single_encoder(self):
        seq_lens = []
        for dataset in self.data["train"].datasets:
            print(dataset[:][self.tensor_names.index("input_ids")].shape)
            train_input_numpy = dataset[:][self.tensor_names.index("input_ids")].numpy()
            seq_lens.extend(np.sum(train_input_numpy != self.processor.tokenizer.pad_token_id, axis=1))
        max_seq_len = dataset[:][self.tensor_names.index("input_ids")].shape[1]
        clipped = np.mean(np.array(seq_lens) == max_seq_len) if seq_lens else 0
        ave_len = np.mean(seq_lens) if seq_lens else 0
        return clipped, ave_len, seq_lens, max_seq_len

    def _calc_length_stats_siamese_encoder(self):
        seq_lens = []
        for dataset in self.data["train"].datasets:
            num_input = dataset[:][self.tensor_names.index("input_ids")].shape[1]
            bs = dataset[:][self.tensor_names.index("input_ids")].shape[0]
            all_input_numpy = (
                dataset[:][self.tensor_names.index("input_ids")].numpy().reshape((bs, -1), order="C")
            )
            input_len = np.sum(all_input_numpy != self.processor.tokenizer.pad_token_id, axis=1) / num_input
            seq_lens.extend(input_len)
        
        max_seq_len = dataset[:][self.tensor_names.index("input_ids")].shape[2]
        clipped = np.mean(np.array(seq_lens) == max_seq_len) if seq_lens else 0
        ave_len = np.mean(seq_lens) if seq_lens else 0

        return clipped, ave_len, seq_lens, max_seq_len
    

    def _calc_length_stats_biencoder(self):
        seq_lens = [[], []]
        for dataset in self.data["train"].datasets:
            query_input_numpy = dataset[:][self.tensor_names.index("query_input_ids")].numpy()
            num_passages = dataset[:][self.tensor_names.index("passage_input_ids")].shape[1]
            bs = dataset[:][self.tensor_names.index("passage_input_ids")].shape[0]
            passage_input_numpy = (
                dataset[:][self.tensor_names.index("passage_input_ids")].numpy().reshape((bs, -1), order="C")
            )
            qlen = np.sum(query_input_numpy != self.processor.query_tokenizer.pad_token_id, axis=1)
            plen = np.sum(passage_input_numpy != self.processor.passage_tokenizer.pad_token_id, axis=1) / num_passages
            seq_lens[0].extend(qlen)
            seq_lens[1].extend(plen)
        q_max_seq_len = dataset[:][self.tensor_names.index("query_input_ids")].shape[1]
        p_max_seq_len = dataset[:][self.tensor_names.index("passage_input_ids")].shape[2]
        clipped_q = np.mean(np.array(seq_lens[0]) == q_max_seq_len) if seq_lens[0] else 0
        ave_len_q = np.mean(seq_lens[0]) if seq_lens[0] else 0
        clipped_p = np.mean(np.array(seq_lens[1]) == p_max_seq_len) if seq_lens[1] else 0
        ave_len_p = np.mean(seq_lens[1]) if seq_lens[1] else 0
        clipped = [clipped_q, clipped_p]
        ave_len = [ave_len_q, ave_len_p]
        max_seq_len = [q_max_seq_len, p_max_seq_len]
        return clipped, ave_len, seq_lens, max_seq_len

    def get_data_loader(self, dataset_name: str):
        """
        Returns data loader for specified split of dataset.

        :param dataset_name: Split of dataset. Either 'train' or 'dev' or 'test'.
        """
        return self.loaders[dataset_name]

    def n_samples(self, dataset_name: str):
        """
        Returns the number of samples in a given dataset.

        :param dataset_name: Split of dataset. Choose from 'train', 'dev' or 'test'.
        """
        return self.counts[dataset_name]


