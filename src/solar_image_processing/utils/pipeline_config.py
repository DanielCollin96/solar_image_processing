from datetime import datetime
from pathlib import Path
from typing import Union

import yaml


class PipelineConfig:
    """
    Read and expose all pipeline parameters from a YAML configuration file.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the YAML configuration file (e.g. ``'configs/pipeline_config.yaml'``).

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the given path.
    ValueError
        If the YAML file cannot be parsed or its top-level type is not a mapping.
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        self._config_path: Path = Path(config_path).resolve()
        if not self._config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}"
            )

        self.content = self._load_yaml()
        self.base_dir = self._resolve_base_dir()

        self.paths = self.content['paths']
        self.start_date = self._parse_date(self.content['start_date'])
        self.end_date = self._parse_date(self.content['end_date'])
        self.channels = self.content['channels']
        self.download_config = self.content['download']
        self.preprocessing_config = self.content['preprocessing']
        self.cropping_config = self.content['cropping']

        self._create_paths()

    def _load_yaml(self) -> dict:
        """
        Load the YAML file and return its contents as a dictionary.

        Returns
        -------
        dict
            Parsed YAML content.

        Raises
        ------
        ValueError
            If the top-level YAML value is not a mapping.
        """
        with open(self._config_path, 'r') as fh:
            content = yaml.safe_load(fh)
        if not isinstance(content, dict):
            raise ValueError(
                f"Configuration file must contain a YAML mapping at the top "
                f"level, but got {type(content).__name__}. "
                f"File: {self._config_path}"
            )
        return content

    def _resolve_base_dir(self) -> Path:
        """
        Determine the project root directory.

        Uses the ``base_dir`` key from the YAML if present; otherwise
        infers it as the parent of the ``configs/`` directory.

        Returns
        -------
        Path
            Absolute path to the project root directory.
        """
        raw_base = self.content.get('base_dir')
        if raw_base is not None:
            return Path(raw_base).resolve()
        # Auto-detect: config file is expected at <project_root>/configs/
        return self._config_path.parent.parent

    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse a date string from the YAML configuration into a datetime.

        Parameters
        ----------
        date_str : str
            ISO-formatted date string. Supported formats:
            ``'YYYY-MM-DD'`` or ``'YYYY-MM-DD HH:MM:SS'``.

        Returns
        -------
        datetime
            Corresponding datetime object.

        Raises
        ------
        ValueError
            If the string does not match any supported format.
        """
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(
            f"Cannot parse date string '{date_str}'. "
            "Supported formats: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
        )

    def _create_paths(self) -> None:
        """
        Resolve all path entries relative to ``base_dir`` and create them.

        Converts each string value in ``self.paths`` to an absolute
        ``Path`` and creates the directory if it does not exist.
        """
        for key in self.paths.keys():
            self.paths[key] = self.base_dir / Path(self.paths[key])
            self.paths[key].mkdir(parents=True, exist_ok=True)
