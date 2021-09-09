"""datasets dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf


# TODO(datasets): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(datasets): BibTeX citation
_CITATION = """
"""


class Datasets(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for datasets dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(datasets): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'raw': tfds.features.Image(shape=(None, None, 3)),
            'qf30': tfds.features.Image(shape=(None, None, 3)),
            # 'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('raw', 'qf30'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(datasets): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    path1 = 'data/raw/'
    path2 = 'data/qf30/'
    sd =  [path1,path2]
    # TODO(datasets): Returns the Dict[split names, Iterator[Key, Example]]
    return [
      tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
           gen_kwargs={"ids":sd},
        ),
    ]

  def _generate_examples(self,ids):
    """Yields examples."""
    import glob
    # TODO(datasets): Yields (key, example) tuples from the dataset
    f1 = list()
    f2 = list()
    for i in range (0,3499):
      f1.append(ids[0]+ str(i+1) +'.png')
    for i in range (0,3499):
      f2.append(ids[1]+ str(i+1) +'.jpeg')
    
    for i in range (0,3499):
      yield i , {
          'raw': f1[i],
          'qf30': f2[i],
      }
      
train_dataset = tfds.load('datasets', 
                    split='train',
                    download=False,
                    )
def f(x):
    # qf30 = tf.image.resize(x['qf30'], [720,1280])
    qf30 = tf.divide(x['qf30'], 255)
    # raw = tf.image.resize(x['raw'], [720,1280])
    raw = tf.divide(x['raw'], 255)
    return qf30,raw
    
train_dataset = train_dataset.map(lambda x: f(x))
