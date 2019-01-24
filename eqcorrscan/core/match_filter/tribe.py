"""
Functions for network matched-filter detection of seismic data.

Designed to cross-correlate templates generated by template_gen function
with data and output the detections.

:copyright:
    EQcorrscan developers.

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import getpass
import glob
import os
import shutil
import tarfile
import tempfile
import logging

import numpy as np
from obspy import Catalog, Stream, read, read_events
from obspy.core.event import Comment, CreationInfo

from eqcorrscan.core.match_filter.template import Template
from eqcorrscan.core.match_filter.party import Party
from eqcorrscan.core.match_filter.helpers import _safemembers, _par_read
from eqcorrscan.core.match_filter.matched_filter import (
    _group_detect, MatchFilterError)
from eqcorrscan.core import template_gen
from eqcorrscan.utils.pre_processing import _check_daylong

Logger = logging.getLogger(__name__)


class Tribe(object):
    """Holder for multiple templates."""

    def __init__(self, templates=None):
        self.templates = []
        if isinstance(templates, Template):
            templates = [templates]
        if templates:
            self.templates.extend(templates)

    def __repr__(self):
        """
        Print information about the tribe.

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='a')])
        >>> print(tribe)
        Tribe of 1 templates
        """
        return 'Tribe of %i templates' % self.__len__()

    def __add__(self, other):
        """
        Add two Tribes or a Tribe and a Template together. '+'

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='a')])
        >>> tribe_ab = tribe + Tribe(templates=[Template(name='b')])
        >>> print(tribe_ab)
        Tribe of 2 templates
        >>> tribe_abc = tribe_ab + Template(name='c')
        >>> print(tribe_abc)
        Tribe of 3 templates
        """
        return self.copy().__iadd__(other)

    def __iadd__(self, other):
        """
        Add in place: '+='

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='a')])
        >>> tribe += Tribe(templates=[Template(name='b')])
        >>> print(tribe)
        Tribe of 2 templates
        >>> tribe += Template(name='c')
        >>> print(tribe)
        Tribe of 3 templates
        """
        if isinstance(other, Tribe):
            self.templates += other.templates
        elif isinstance(other, Template):
            self.templates.append(other)
        else:
            raise TypeError('Must be either Template or Tribe')
        return self

    def __eq__(self, other):
        """
        Test for equality. Rich comparison operator '=='

        .. rubric:: Example

        >>> tribe_a = Tribe(templates=[Template(name='a')])
        >>> tribe_b = Tribe(templates=[Template(name='b')])
        >>> tribe_a == tribe_b
        False
        >>> tribe_a == tribe_a
        True
        """
        if self.sort().templates != other.sort().templates:
            return False
        return True

    def __ne__(self, other):
        """
        Test for inequality. Rich comparison operator '!='

        .. rubric:: Example

        >>> tribe_a = Tribe(templates=[Template(name='a')])
        >>> tribe_b = Tribe(templates=[Template(name='b')])
        >>> tribe_a != tribe_b
        True
        >>> tribe_a != tribe_a
        False
        """
        return not self.__eq__(other)

    def __len__(self):
        """
        Number of Templates in Tribe. len(tribe)

        .. rubric:: Example

        >>> tribe_a = Tribe(templates=[Template(name='a')])
        >>> len(tribe_a)
        1
        """
        return len(self.templates)

    def __iter__(self):
        """
        Iterator for the Tribe.
        """
        return list(self.templates).__iter__()

    def __getitem__(self, index):
        """
        Support slicing to get Templates from Tribe.

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='a'), Template(name='b'),
        ...                          Template(name='c')])
        >>> tribe[1] # doctest: +NORMALIZE_WHITESPACE
        Template b:
         0 channels;
         lowcut: None Hz;
         highcut: None Hz;
         sampling rate None Hz;
         filter order: None;
         process length: None s
        >>> tribe[0:2]
        Tribe of 2 templates
        """
        if isinstance(index, slice):
            return self.__class__(templates=self.templates.__getitem__(index))
        elif isinstance(index, int):
            return self.templates.__getitem__(index)
        else:
            _index = [i for i, t in enumerate(self.templates)
                      if t.name == index]
            try:
                return self.templates.__getitem__(_index[0])
            except IndexError:
                Logger.warning('Template: %s not in tribe' % index)
                return []

    def sort(self):
        """
        Sort the tribe, sorts by template name.

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='c'), Template(name='b'),
        ...                          Template(name='a')])
        >>> tribe.sort()
        Tribe of 3 templates
        >>> tribe[0] # doctest: +NORMALIZE_WHITESPACE
        Template a:
         0 channels;
         lowcut: None Hz;
         highcut: None Hz;
         sampling rate None Hz;
         filter order: None;
         process length: None s
        """
        self.templates = sorted(self.templates, key=lambda x: x.name)
        return self

    def select(self, template_name):
        """
        Select a particular template from the tribe.

        :type template_name: str
        :param template_name: Template name to look-up
        :return: Template

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='c'), Template(name='b'),
        ...                          Template(name='a')])
        >>> tribe.select('b') # doctest: +NORMALIZE_WHITESPACE
        Template b:
         0 channels;
         lowcut: None Hz;
         highcut: None Hz;
         sampling rate None Hz;
         filter order: None;
         process length: None s
        """
        return [t for t in self.templates if t.name == template_name][0]

    def remove(self, template):
        """
        Remove a template from the tribe.

        :type template: :class:`eqcorrscan.core.match_filter.Template`
        :param template: Template to remove from tribe

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='c'), Template(name='b'),
        ...                          Template(name='a')])
        >>> tribe.remove(tribe.templates[0])
        Tribe of 2 templates
        """
        self.templates = [t for t in self.templates if t != template]
        return self

    def copy(self):
        """
        Copy the Tribe.

        .. rubric:: Example

        >>> tribe_a = Tribe(templates=[Template(name='a')])
        >>> tribe_b = tribe_a.copy()
        >>> tribe_a == tribe_b
        True
        """
        return copy.deepcopy(self)

    def write(self, filename, compress=True, catalog_format="QUAKEML"):
        """
        Write the tribe to a file using tar archive formatting.

        :type filename: str
        :param filename:
            Filename to write to, if it exists it will be appended to.
        :type compress: bool
        :param compress:
            Whether to compress the tar archive or not, if False then will
            just be files in a folder.
        :type catalog_format: str
        :param catalog_format:
            What format to write the detection-catalog with. Only Nordic,
            SC3ML, QUAKEML are supported. Note that not all information is
            written for all formats (QUAKEML is the most complete, but is
            slow for IO).

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='c', st=read())])
        >>> tribe.write('test_tribe')
        Tribe of 1 templates
        """
        from eqcorrscan.core.match_filter import CAT_EXT_MAP

        if catalog_format not in CAT_EXT_MAP.keys():
            raise TypeError("{0} is not supported".format(catalog_format))
        if not os.path.isdir(filename):
            os.makedirs(filename)
        self._par_write(filename)
        tribe_cat = Catalog()
        for t in self.templates:
            if t.event is not None:
                tribe_cat.append(t.event)
        if len(tribe_cat) > 0:
            tribe_cat.write(
                os.path.join(filename, 'tribe_cat.{0}'.format(
                    CAT_EXT_MAP[catalog_format])), format=catalog_format)
        for template in self.templates:
            template.st.write(filename + '/' + template.name + '.ms',
                              format='MSEED')
        if compress:
            with tarfile.open(filename + '.tgz', "w:gz") as tar:
                tar.add(filename, arcname=os.path.basename(filename))
            shutil.rmtree(filename)
        return self

    def _par_write(self, dirname):
        """
        Internal write function to write a formatted parameter file.

        :type dirname: str
        :param dirname: Directory to write the parameter file to.
        """
        filename = dirname + '/' + 'template_parameters.csv'
        with open(filename, 'w') as parfile:
            for template in self.templates:
                for key in template.__dict__.keys():
                    if key not in ['st', 'event']:
                        parfile.write(key + ': ' +
                                      str(template.__dict__[key]) + ', ')
                parfile.write('\n')
        return self

    def read(self, filename):
        """
        Read a tribe of templates from a tar formatted file.

        :type filename: str
        :param filename: File to read templates from.

        .. rubric:: Example

        >>> tribe = Tribe(templates=[Template(name='c', st=read())])
        >>> tribe.write('test_tribe')
        Tribe of 1 templates
        >>> tribe_back = Tribe().read('test_tribe.tgz')
        >>> tribe_back == tribe
        True
        """
        with tarfile.open(filename, "r:*") as arc:
            temp_dir = tempfile.mkdtemp()
            arc.extractall(path=temp_dir, members=_safemembers(arc))
            tribe_dir = glob.glob(temp_dir + os.sep + '*')[0]
            self._read_from_folder(dirname=tribe_dir)
        shutil.rmtree(temp_dir)
        return self

    def _read_from_folder(self, dirname):
        """
        Internal folder reader.

        :type dirname: str
        :param dirname: Folder to read from.
        """
        templates = _par_read(dirname=dirname, compressed=False)
        t_files = glob.glob(dirname + os.sep + '*.ms')
        tribe_cat_file = glob.glob(os.path.join(dirname, "tribe_cat.*"))
        if len(tribe_cat_file) != 0:
            tribe_cat = read_events(tribe_cat_file[0])
        else:
            tribe_cat = Catalog()
        previous_template_names = [t.name for t in self.templates]
        for template in templates:
            if template.name in previous_template_names:
                # Don't read in for templates that we already have.
                continue
            for event in tribe_cat:
                for comment in event.comments:
                    if comment.text == 'eqcorrscan_template_' + template.name:
                        template.event = event
            t_file = [t for t in t_files
                      if t.split(os.sep)[-1] == template.name + '.ms']
            if len(t_file) == 0:
                Logger.error('No waveform for template: ' + template.name)
                templates.remove(template)
                continue
            elif len(t_file) > 1:
                Logger.warning('Multiple waveforms found, using: ' + t_file[0])
            template.st = read(t_file[0])
        self.templates.extend(templates)
        return

    def cluster(self, method, **kwargs):
        """
        Cluster the tribe.

        Cluster templates within a tribe: returns multiple tribes each of
        which could be stacked.

        :type method: str
        :param method:
            Method of stacking, see :mod:`eqcorrscan.utils.clustering`

        :return: List of tribes.

        .. rubric:: Example


        """
        from eqcorrscan.utils import clustering
        tribes = []
        func = getattr(clustering, method)
        if method in ['space_cluster', 'space_time_cluster']:
            cat = Catalog([t.event for t in self.templates])
            groups = func(cat, **kwargs)
            for group in groups:
                new_tribe = Tribe()
                for event in group:
                    new_tribe.templates.extend([t for t in self.templates
                                                if t.event == event])
                tribes.append(new_tribe)
        return tribes

    def detect(self, stream, threshold, threshold_type, trig_int, plotvar,
               daylong=False, parallel_process=True, xcorr_func=None,
               concurrency=None, cores=None, ignore_length=False,
               group_size=None, overlap="calculate", full_peaks=False,
               save_progress=False, process_cores=None, **kwargs):
        """
        Detect using a Tribe of templates within a continuous stream.

        :type stream: `obspy.core.stream.Stream`
        :param stream: Continuous data to detect within using the Template.
        :type threshold: float
        :param threshold:
            Threshold level, if using `threshold_type='MAD'` then this will be
            the multiple of the median absolute deviation.
        :type threshold_type: str
        :param threshold_type:
            The type of threshold to be used, can be MAD, absolute or
            av_chan_corr.  See Note on thresholding below.
        :type trig_int: float
        :param trig_int:
            Minimum gap between detections in seconds. If multiple detections
            occur within trig_int of one-another, the one with the highest
            cross-correlation sum will be selected.
        :type plotvar: bool
        :param plotvar:
            Turn plotting on or off, see warning about plotting below
        :type daylong: bool
        :param daylong:
            Set to True to use the
            :func:`eqcorrscan.utils.pre_processing.dayproc` routine, which
            preforms additional checks and is more efficient for day-long data
            over other methods.
        :type parallel_process: bool
        :param parallel_process:
        :type xcorr_func: str or callable
        :param xcorr_func:
            A str of a registered xcorr function or a callable for implementing
            a custom xcorr function. For more information see:
            :func:`eqcorrscan.utils.correlate.register_array_xcorr`
        :type concurrency: str
        :param concurrency:
            The type of concurrency to apply to the xcorr function. Options are
            'multithread', 'multiprocess', 'concurrent'. For more details see
            :func:`eqcorrscan.utils.correlate.get_stream_xcorr`
        :type cores: int
        :param cores: Number of workers for procesisng and detection.
        :type ignore_length: bool
        :param ignore_length:
            If using daylong=True, then dayproc will try check that the data
            are there for at least 80% of the day, if you don't want this check
            (which will raise an error if too much data are missing) then set
            ignore_length=True.  This is not recommended!
        :type group_size: int
        :param group_size:
            Maximum number of templates to run at once, use to reduce memory
            consumption, if unset will use all templates.
        :type overlap: float
        :param overlap:
            Either None, "calculate" or a float of number of seconds to
            overlap detection streams by.  This is to counter the effects of
            the delay-and-stack in calculating cross-correlation sums. Setting
            overlap = "calculate" will work out the appropriate overlap based
            on the maximum lags within templates.
        :type full_peaks: bool
        :param full_peaks: See `eqcorrscan.utils.findpeak.find_peaks2_short`
        :type save_progress: bool
        :param save_progress:
            Whether to save the resulting party at every data step or not.
            Useful for long-running processes.
        :type process_cores: int
        :param process_cores:
            Number of processes to use for pre-processing (if different to
            `cores`).

        :return:
            :class:`eqcorrscan.core.match_filter.Party` of Families of
            detections.

        .. Note::
            When using the "fftw" correlation backend the length of the fft
            can be set. See :mod:`eqcorrscan.utils.correlate` for more info.

        .. Note::
            `stream` must not be pre-processed. If your data contain gaps
            you should *NOT* fill those gaps before using this method.
            The pre-process functions (called within) will fill the gaps
            internally prior to processing, process the data, then re-fill
            the gaps with zeros to ensure correlations are not incorrectly
            calculated within gaps. If your data have gaps you should pass a
            merged stream without the `fill_value` argument
            (e.g.: `stream = stream.merge()`).

        .. Note::
            Detections are not corrected for `pre-pick`, the
            detection.detect_time corresponds to the beginning of the earliest
            template channel at detection.

        .. warning::
            Picks included in the output Party.get_catalog() will not be
            corrected for pre-picks in the template.

        .. note::
            **Data overlap:**

            Internally this routine shifts and trims the data according to the
            offsets in the template (e.g. if trace 2 starts 2 seconds after
            trace 1 in the template then the continuous data will be shifted
            by 2 seconds to align peak correlations prior to summing).
            Because of this, detections at the start and end of continuous
            data streams **may be missed**.  The maximum time-period that
            might be missing detections is the maximum offset in the template.

            To work around this, if you are conducting matched-filter
            detections through long-duration continuous data, we suggest
            using some overlap (a few seconds, on the order of the maximum
            offset in the templates) in the continuous data.  You will then
            need to post-process the detections (which should be done anyway
            to remove duplicates).  See below note for how `overlap` argument
            affects data internally if `stream` is longer than the processing
            length.

        .. Note::
            If `stream` is longer than processing length, this routine will
            ensure that data overlap between loops, which will lead to no
            missed detections at data start-stop points (see above note).
            This will result in end-time not being strictly
            honoured, so detections may occur after the end-time set.  This is
            because data must be run in the correct process-length.

        .. note::
            **Thresholding:**

            **MAD** threshold is calculated as the:

            .. math::

                threshold {\\times} (median(abs(cccsum)))

            where :math:`cccsum` is the cross-correlation sum for a given
            template.

            **absolute** threshold is a true absolute threshold based on the
            cccsum value.

            **av_chan_corr** is based on the mean values of single-channel
            cross-correlations assuming all data are present as required for
            the template, e.g:

            .. math::

                av\_chan\_corr\_thresh=threshold \\times (cccsum /
                len(template))

            where :math:`template` is a single template from the input and the
            length is the number of channels within this template.
        """
        party = Party()
        template_groups = []
        for master in self.templates:
            for group in template_groups:
                if master in group:
                    break
            else:
                new_group = [master]
                for slave in self.templates:
                    if master.same_processing(slave) and master != slave:
                        new_group.append(slave)
                template_groups.append(new_group)
        # template_groups will contain an empty first list
        for group in template_groups:
            if len(group) == 0:
                template_groups.remove(group)
        # now we can compute the detections for each group
        for group in template_groups:
            group_party = _group_detect(
                templates=group, stream=stream.copy(), threshold=threshold,
                threshold_type=threshold_type, trig_int=trig_int,
                plotvar=plotvar, group_size=group_size, pre_processed=False,
                daylong=daylong, parallel_process=parallel_process,
                xcorr_func=xcorr_func, concurrency=concurrency, cores=cores,
                ignore_length=ignore_length, overlap=overlap,
                full_peaks=full_peaks, process_cores=process_cores, **kwargs)
            party += group_party
            if save_progress:
                party.write("eqcorrscan_temporary_party")
        if len(party) > 0:
            for family in party:
                if family is not None:
                    family.detections = family._uniq().detections
        return party

    def client_detect(self, client, starttime, endtime, threshold,
                      threshold_type, trig_int, plotvar, min_gap=None,
                      daylong=False, parallel_process=True, xcorr_func=None,
                      concurrency=None, cores=None, ignore_length=False,
                      group_size=None, return_stream=False,
                      full_peaks=False, save_progress=False,
                      process_cores=None, retries=3, **kwargs):
        """
        Detect using a Tribe of templates within a continuous stream.

        :type client: `obspy.clients.*.Client`
        :param client: Any obspy client with a dataselect service.
        :type starttime: :class:`obspy.core.UTCDateTime`
        :param starttime: Start-time for detections.
        :type endtime: :class:`obspy.core.UTCDateTime`
        :param endtime: End-time for detections
        :type threshold: float
        :param threshold:
            Threshold level, if using `threshold_type='MAD'` then this will be
            the multiple of the median absolute deviation.
        :type threshold_type: str
        :param threshold_type:
            The type of threshold to be used, can be MAD, absolute or
            av_chan_corr.  See Note on thresholding below.
        :type trig_int: float
        :param trig_int:
            Minimum gap between detections in seconds. If multiple detections
            occur within trig_int of one-another, the one with the highest
            cross-correlation sum will be selected.
        :type plotvar: bool
        :param plotvar:
            Turn plotting on or off, see warning about plotting below
        :type min_gap: float
        :param min_gap:
            Minimum gap allowed in data - use to remove traces with known
            issues
        :type daylong: bool
        :param daylong:
            Set to True to use the
            :func:`eqcorrscan.utils.pre_processing.dayproc` routine, which
            preforms additional checks and is more efficient for day-long data
            over other methods.
        :type parallel_process: bool
        :param parallel_process:
        :type xcorr_func: str or callable
        :param xcorr_func:
            A str of a registered xcorr function or a callable for implementing
            a custom xcorr function. For more information see:
            :func:`eqcorrscan.utils.correlate.register_array_xcorr`
        :type concurrency: str
        :param concurrency:
            The type of concurrency to apply to the xcorr function. Options are
            'multithread', 'multiprocess', 'concurrent'. For more details see
            :func:`eqcorrscan.utils.correlate.get_stream_xcorr`
        :type cores: int
        :param cores: Number of workers for processing and detection.
        :type ignore_length: bool
        :param ignore_length:
            If using daylong=True, then dayproc will try check that the data
            are there for at least 80% of the day, if you don't want this check
            (which will raise an error if too much data are missing) then set
            ignore_length=True.  This is not recommended!
        :type group_size: int
        :param group_size:
            Maximum number of templates to run at once, use to reduce memory
            consumption, if unset will use all templates.
        :type full_peaks: bool
        :param full_peaks: See `eqcorrscan.utils.findpeaks.find_peaks2_short`
        :type save_progress: bool
        :param save_progress:
            Whether to save the resulting party at every data step or not.
            Useful for long-running processes.
        :type process_cores: int
        :param process_cores:
            Number of processes to use for pre-processing (if different to
            `cores`).
        :type return_stream: bool
        :param return_stream:
            Whether to also output the stream downloaded, useful if you plan
            to use the stream for something else, e.g. lag_calc.
        :type retries: int
        :param retries:
            Number of attempts allowed for downloading - allows for transient
            server issues.

        :return:
            :class:`eqcorrscan.core.match_filter.Party` of Families of
            detections.


        .. Note::
            When using the "fftw" correlation backend the length of the fft
            can be set. See :mod:`eqcorrscan.utils.correlate` for more info.

        .. Note::
            Detections are not corrected for `pre-pick`, the
            detection.detect_time corresponds to the beginning of the earliest
            template channel at detection.

        .. warning::
            Picks included in the output Party.get_catalog() will not be
            corrected for pre-picks in the template.

        .. Note::
            Ensures that data overlap between loops, which will lead to no
            missed detections at data start-stop points (see note for
            :meth:`eqcorrscan.core.match_filter.Tribe.detect` method).
            This will result in end-time not being strictly
            honoured, so detections may occur after the end-time set.  This is
            because data must be run in the correct process-length.

        .. warning::
            Plotting within the match-filter routine uses the Agg backend
            with interactive plotting turned off.  This is because the function
            is designed to work in bulk.  If you wish to turn interactive
            plotting on you must import matplotlib in your script first,
            when you then import match_filter you will get the warning that
            this call to matplotlib has no effect, which will mean that
            match_filter has not changed the plotting behaviour.

        .. note::
            **Thresholding:**

            **MAD** threshold is calculated as the:

            .. math::

                threshold {\\times} (median(abs(cccsum)))

            where :math:`cccsum` is the cross-correlation sum for a given
            template.

            **absolute** threshold is a true absolute threshold based on the
            cccsum value.

            **av_chan_corr** is based on the mean values of single-channel
            cross-correlations assuming all data are present as required for
            the template, e.g:

            .. math::

                av\_chan\_corr\_thresh=threshold \\times (cccsum /
                len(template))

            where :math:`template` is a single template from the input and the
            length is the number of channels within this template.
        """
        party = Party()
        buff = 300
        # Apply a buffer, often data downloaded is not the correct length
        data_length = max([t.process_length for t in self.templates])
        pad = 0
        for template in self.templates:
            max_delay = (template.st.sort(['starttime'])[-1].stats.starttime -
                         template.st.sort(['starttime'])[0].stats.starttime)
            if max_delay > pad:
                pad = max_delay
        download_groups = int(endtime - starttime) / data_length
        template_channel_ids = []
        for template in self.templates:
            for tr in template.st:
                if tr.stats.network not in [None, '']:
                    chan_id = (tr.stats.network,)
                else:
                    chan_id = ('*',)
                if tr.stats.station not in [None, '']:
                    chan_id += (tr.stats.station,)
                else:
                    chan_id += ('*',)
                if tr.stats.location not in [None, '']:
                    chan_id += (tr.stats.location,)
                else:
                    chan_id += ('*',)
                if tr.stats.channel not in [None, '']:
                    if len(tr.stats.channel) == 2:
                        chan_id += (tr.stats.channel[0] + '?' +
                                    tr.stats.channel[-1],)
                    else:
                        chan_id += (tr.stats.channel,)
                else:
                    chan_id += ('*',)
                template_channel_ids.append(chan_id)
        template_channel_ids = list(set(template_channel_ids))
        if return_stream:
            stream = Stream()
        if int(download_groups) < download_groups:
            download_groups = int(download_groups) + 1
        else:
            download_groups = int(download_groups)
        for i in range(download_groups):
            bulk_info = []
            for chan_id in template_channel_ids:
                bulk_info.append((
                    chan_id[0], chan_id[1], chan_id[2], chan_id[3],
                    starttime + (i * data_length) - (pad + buff),
                    starttime + ((i + 1) * data_length) + (pad + buff)))
            for retry_attempt in range(retries):
                try:
                    Logger.info("Downloading data")
                    st = client.get_waveforms_bulk(bulk_info)
                    Logger.info(
                        "Downloaded data for {0} traces".format(len(st)))
                    break
                except Exception as e:
                    Logger.error(e)
                    continue
            else:
                raise MatchFilterError(
                    "Could not download data after {0} attempts".format(
                        retries))
            # Get gaps and remove traces as necessary
            if min_gap:
                gaps = st.get_gaps(min_gap=min_gap)
                if len(gaps) > 0:
                    Logger.warning("Large gaps in downloaded data")
                    st.merge()
                    gappy_channels = list(
                        set([(gap[0], gap[1], gap[2], gap[3])
                             for gap in gaps]))
                    _st = Stream()
                    for tr in st:
                        tr_stats = (tr.stats.network, tr.stats.station,
                                    tr.stats.location, tr.stats.channel)
                        if tr_stats in gappy_channels:
                            Logger.warning(
                                "Removing gappy channel: {0}".format(tr))
                        else:
                            _st += tr
                    st = _st
                    st.split()
            st.merge()
            st.trim(starttime=starttime + (i * data_length) - pad,
                    endtime=starttime + ((i + 1) * data_length) + pad)
            for tr in st:
                if not _check_daylong(tr):
                    st.remove(tr)
                    Logger.warning(
                        "{0} contains more zeros than non-zero, "
                        "removed".format(tr.id))
            for tr in st:
                if tr.stats.endtime - tr.stats.starttime < \
                   0.8 * data_length:
                    st.remove(tr)
                    Logger.warning(
                        "{0} is less than 80% of the required length"
                        ", removed".format(tr.id))
            if return_stream:
                stream += st
            try:
                party += self.detect(
                    stream=st, threshold=threshold,
                    threshold_type=threshold_type, trig_int=trig_int,
                    plotvar=plotvar, daylong=daylong,
                    parallel_process=parallel_process, xcorr_func=xcorr_func,
                    concurrency=concurrency, cores=cores,
                    ignore_length=ignore_length, group_size=group_size,
                    overlap=None, full_peaks=full_peaks,
                    process_cores=process_cores, **kwargs)
                if save_progress:
                    party.write("eqcorrscan_temporary_party")
            except Exception as e:
                Logger.critical(
                    'Error, routine incomplete, returning incomplete Party')
                Logger.error('Error: {0}'.format(e))
                if return_stream:
                    return party, stream
                else:
                    return party
        for family in party:
            if family is not None:
                family.detections = family._uniq().detections
        if return_stream:
            return party, stream
        else:
            return party

    def construct(self, method, lowcut, highcut, samp_rate, filt_order,
                  prepick, save_progress=False, **kwargs):
        """
        Generate a Tribe of Templates.

        See :mod:`eqcorrscan.core.template_gen` for available methods.

        :param method: Method of Tribe generation.
        :param kwargs: Arguments for the given method.
        :type lowcut: float
        :param lowcut:
            Low cut (Hz), if set to None will not apply a lowcut
        :type highcut: float
        :param highcut:
            High cut (Hz), if set to None will not apply a highcut.
        :type samp_rate: float
        :param samp_rate:
            New sampling rate in Hz.
        :type filt_order: int
        :param filt_order:
            Filter level (number of corners).
        :type prepick: float
        :param prepick: Pre-pick time in seconds
        :type save_progress: bool
        :param save_progress:
            Whether to save the resulting party at every data step or not.
            Useful for long-running processes.

        .. Note::
            Methods: `from_contbase`, `from_sfile` and `from_sac` are not
            supported by Tribe.construct and must use Template.construct.

        .. Note::
            The Method `multi_template_gen` is not supported because the
            processing parameters for the stream are not known. Use
            `from_meta_file` instead.

        .. Note:: Templates will be named according to their start-time.
        """
        templates, catalog, process_lengths = template_gen.template_gen(
            method=method, lowcut=lowcut, highcut=highcut,
            filt_order=filt_order, samp_rate=samp_rate, prepick=prepick,
            return_event=True, save_progress=save_progress, **kwargs)
        for template, event, process_len in zip(templates, catalog,
                                                process_lengths):
            t = Template()
            for tr in template:
                if not np.any(tr.data.astype(np.float16)):
                    Logger.warning('Data are zero in float16, missing data,'
                                   ' will not use: {0}'.format(tr.id))
                    template.remove(tr)
            if len(template) == 0:
                Logger.error('Empty Template')
                continue
            t.st = template
            t.name = template.sort(['starttime'])[0]. \
                stats.starttime.strftime('%Y_%m_%dt%H_%M_%S')
            t.lowcut = lowcut
            t.highcut = highcut
            t.filt_order = filt_order
            t.samp_rate = samp_rate
            t.process_length = process_len
            t.prepick = prepick
            event.comments.append(Comment(
                text="eqcorrscan_template_" + t.name,
                creation_info=CreationInfo(agency='eqcorrscan',
                                           author=getpass.getuser())))
            t.event = event
            self.templates.append(t)
        return self


def read_tribe(fname):
    """
    Read a Tribe of templates from a tar archive.

    :param fname: Filename to read from
    :return: :class:`eqcorrscan.core.match_filter.Tribe`
    """
    tribe = Tribe()
    tribe.read(filename=fname)
    return tribe


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # List files to be removed after doctest
    cleanup = ['test_tribe.tgz']
    for f in cleanup:
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
