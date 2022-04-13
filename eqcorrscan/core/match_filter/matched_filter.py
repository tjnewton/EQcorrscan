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
import logging
from timeit import default_timer

import numpy as np
from obspy import Catalog, UTCDateTime, Stream

from eqcorrscan.core.match_filter.helpers import (
    _spike_test, extract_from_stream)

from eqcorrscan.utils.correlate import get_stream_xcorr
from eqcorrscan.utils.findpeaks import multi_find_peaks
from eqcorrscan.utils.pre_processing import (
    dayproc, shortproc, _prep_data_for_correlation)

Logger = logging.getLogger(__name__)


class MatchFilterError(Exception):
    """
    Default error for match-filter errors.
    """

    def __init__(self, value):
        """
        Raise error.

        .. rubric:: Example

        >>> MatchFilterError('This raises an error')
        This raises an error
        """
        self.value = value

    def __repr__(self):
        """
        Print the value of the error.

        .. rubric:: Example
        >>> print(MatchFilterError('Error').__repr__())
        Error
        """
        return self.value

    def __str__(self):
        """
        Print the error in a pretty way.

        .. rubric:: Example
        >>> print(MatchFilterError('Error'))
        Error
        """
        return self.value


def _group_detect(templates, stream, threshold, threshold_type, trig_int,
                  plot=False, plotdir=None, group_size=None,
                  pre_processed=False, daylong=False, parallel_process=True,
                  xcorr_func=None, concurrency=None, cores=None,
                  ignore_length=False, ignore_bad_data=False,
                  overlap="calculate", full_peaks=False, process_cores=None,
                  **kwargs):
    """
    Pre-process and compute detections for a group of templates.

    Will process the stream object, so if running in a loop, you will want
    to copy the stream before passing it to this function.

    :type templates: list
    :param templates: List of :class:`eqcorrscan.core.match_filter.Template`
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
        Minimum gap between detections from one template in seconds.
        If multiple detections occur within trig_int of one-another, the one
        with the highest cross-correlation sum will be selected.
    :type plot: bool
    :param plot:
        Turn plotting on or off.
    :type plotdir: str
    :param plotdir:
        The path to save plots to. If `plotdir=None` (default) then the
        figure will be shown on screen.
    :type group_size: int
    :param group_size:
        Maximum number of templates to run at once, use to reduce memory
        consumption, if unset will use all templates.
    :type pre_processed: bool
    :param pre_processed:
        Set to True if `stream` has already undergone processing, in this
        case eqcorrscan will only check that the sampling rate is correct.
        Defaults to False, which will use the
        :mod:`eqcorrscan.utils.pre_processing` routines to resample and
        filter the continuous data.
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
        a custom xcorr function. For more details see:
        :func:`eqcorrscan.utils.correlate.register_array_xcorr`
    :type concurrency: str
    :param concurrency:
        The type of concurrency to apply to the xcorr function. Options are
        'multithread', 'multiprocess', 'concurrent'. For more details see
        :func:`eqcorrscan.utils.correlate.get_stream_xcorr`
    :type cores: int
    :param cores: Number of workers for processing and correlation.
    :type ignore_length: bool
    :param ignore_length:
        If using daylong=True, then dayproc will try check that the data
        are there for at least 80% of the day, if you don't want this check
        (which will raise an error if too much data are missing) then set
        ignore_length=True.  This is not recommended!
    :type overlap: float
    :param overlap:
        Either None, "calculate" or a float of number of seconds to
        overlap detection streams by.  This is to counter the effects of
        the delay-and-stack in calculating cross-correlation sums. Setting
        overlap = "calculate" will work out the appropriate overlap based
        on the maximum lags within templates.
    :type full_peaks: bool
    :param full_peaks: See `eqcorrscan.utils.findpeaks.find_peaks_compiled`
    :type process_cores: int
    :param process_cores:
        Number of processes to use for pre-processing (if different to
        `cores`).

    :return:
        :class:`eqcorrscan.core.match_filter.Party` of families of detections.
    """
    from eqcorrscan.core.match_filter.party import Party
    from eqcorrscan.core.match_filter.family import Family

    master = templates[0]
    peak_cores = kwargs.get('peak_cores', process_cores)
    kwargs.update(dict(peak_cores=peak_cores))
    # Check that they are all processed the same.
    lap = 0.0
    for template in templates:
        starts = [t.stats.starttime for t in template.st.sort(['starttime'])]
        if starts[-1] - starts[0] > lap:
            lap = starts[-1] - starts[0]
        if not template.same_processing(master):
            raise MatchFilterError('Templates must be processed the same.')
    if overlap is None:
        overlap = 0.0
    elif not isinstance(overlap, float) and str(overlap) == str("calculate"):
        overlap = lap
    elif not isinstance(overlap, float):
        raise NotImplementedError(
            "%s is not a recognised overlap type" % str(overlap))
    if overlap >= master.process_length:
        Logger.warning(
                f"Overlap of {overlap} s is greater than process "
                f"length ({master.process_length} s), ignoring overlap")
        overlap = 0
    if not pre_processed:
        if process_cores is None:
            process_cores = cores
        streams = _group_process(
            template_group=templates, parallel=parallel_process,
            cores=process_cores, stream=stream, daylong=daylong,
            ignore_length=ignore_length, ignore_bad_data=ignore_bad_data,
            overlap=overlap)
        for _st in streams:
            Logger.debug(f"Processed stream:\n{_st.__str__(extended=True)}")
    else:
        Logger.warning('Not performing any processing on the continuous data.')
        streams = [stream]
    detections = []
    party = Party()
    if group_size is not None:
        n_groups = int(len(templates) / group_size)
        if n_groups * group_size < len(templates):
            n_groups += 1
    else:
        n_groups = 1
    for st_chunk in streams:
        chunk_start, chunk_end = (min(tr.stats.starttime for tr in st_chunk),
                                  max(tr.stats.endtime for tr in st_chunk))
        Logger.info(
            f'Computing detections between {chunk_start} and {chunk_end}')
        st_chunk.trim(starttime=chunk_start, endtime=chunk_end)
        for tr in st_chunk:
            if len(tr) > len(st_chunk[0]):
                tr.data = tr.data[0:len(st_chunk[0])]
        for i in range(n_groups):
            if group_size is not None:
                end_group = (i + 1) * group_size
                start_group = i * group_size
                if i == n_groups:
                    end_group = len(templates)
            else:
                end_group = len(templates)
                start_group = 0
            template_group = [t for t in templates[start_group: end_group]]
            detections += match_filter(
                template_names=[t.name for t in template_group],
                template_list=[t.st for t in template_group], st=st_chunk,
                xcorr_func=xcorr_func, concurrency=concurrency,
                threshold=threshold, threshold_type=threshold_type,
                trig_int=trig_int, plot=plot, plotdir=plotdir, cores=cores,
                full_peaks=full_peaks, **kwargs)
            for template in template_group:
                family = Family(template=template, detections=[])
                for detection in detections:
                    if detection.template_name == template.name:
                        for pick in detection.event.picks:
                            pick.time += template.prepick
                        for origin in detection.event.origins:
                            origin.time += template.prepick
                        family.detections.append(detection)
                party += family
    return party


def _group_process(template_group, parallel, cores, stream, daylong,
                   ignore_length, ignore_bad_data, overlap):
    """
    Process data into chunks based on template processing length.

    Templates in template_group must all have the same processing parameters.

    :type template_group: list
    :param template_group: List of Templates.
    :type parallel: bool
    :param parallel: Whether to use parallel processing or not
    :type cores: int
    :param cores: Number of cores to use, can be False to use all available.
    :type stream: :class:`obspy.core.stream.Stream`
    :param stream: Stream to process, will be left intact.
    :type daylong: bool
    :param daylong: Whether to enforce day-length files or not.
    :type ignore_length: bool
    :param ignore_length:
        If using daylong=True, then dayproc will try check that the data
        are there for at least 80% of the day, if you don't want this check
        (which will raise an error if too much data are missing) then set
        ignore_length=True.  This is not recommended!
    :type ignore_bad_data: bool
    :param ignore_bad_data:
        If False (default), errors will be raised if data are excessively
        gappy or are mostly zeros. If True then no error will be raised, but
        an empty trace will be returned.
    :type overlap: float
    :param overlap: Number of seconds to overlap chunks by.

    :return: list of processed streams.
    """
    master = template_group[0]
    processed_streams = []
    kwargs = {
        'filt_order': master.filt_order,
        'highcut': master.highcut, 'lowcut': master.lowcut,
        'samp_rate': master.samp_rate, 'parallel': parallel,
        'num_cores': cores, 'ignore_length': ignore_length,
        'ignore_bad_data': ignore_bad_data}
    # Processing always needs to be run to account for gaps - pre-process will
    # check whether filtering and resampling needs to be done.
    process_length = master.process_length
    if daylong:
        if not master.process_length == 86400:
            Logger.warning(
                'Processing day-long data, but template was cut from %i s long'
                ' data, will reduce correlations' % master.process_length)
        func = dayproc
        process_length = 86400
        # Check that data all start on the same day, otherwise strange
        # things will happen...
        starttimes = [tr.stats.starttime.date for tr in stream]
        if not len(list(set(starttimes))) == 1:
            Logger.warning('Data start on different days, setting to last day')
            starttime = UTCDateTime(
                stream.sort(['starttime'])[-1].stats.starttime.date)
        else:
            starttime = stream.sort(['starttime'])[0].stats.starttime
    else:
        # We want to use shortproc to allow overlaps
        func = shortproc
        starttime = stream.sort(['starttime'])[0].stats.starttime
    endtime = stream.sort(['endtime'])[-1].stats.endtime
    data_len_samps = round((endtime - starttime) * master.samp_rate) + 1
    assert overlap < process_length, "Overlap must be less than process length"
    chunk_len_samps = (process_length - overlap) * master.samp_rate
    n_chunks = int(data_len_samps // chunk_len_samps)
    Logger.info(f"Splitting these data in {n_chunks} chunks")
    if n_chunks == 0:
        Logger.error('Data must be process_length or longer, not computing')
    _endtime = starttime
    for i in range(n_chunks):
        kwargs.update(
            {'starttime': starttime + (i * (process_length - overlap))})
        if not daylong:
            _endtime = kwargs['starttime'] + process_length
            kwargs.update({'endtime': _endtime})
        else:
            _endtime = kwargs['starttime'] + 86400
        chunk_stream = stream.slice(starttime=kwargs['starttime'],
                                    endtime=_endtime).copy()
        Logger.debug(f"Processing chunk {i} between {kwargs['starttime']} "
                     f"and {_endtime}")
        if len(chunk_stream) == 0:
            Logger.warning(
                f"No data between {kwargs['starttime']} and {_endtime}")
            continue
        for tr in chunk_stream:
            tr.data = tr.data[0:int(
                process_length * tr.stats.sampling_rate)]
        _chunk_stream_lengths = {
            tr.id: tr.stats.endtime - tr.stats.starttime
            for tr in chunk_stream}
        for tr_id, chunk_length in _chunk_stream_lengths.items():
            # Remove traces that are too short.
            if not ignore_length and chunk_length <= .8 * process_length:
                tr = chunk_stream.select(id=tr_id)[0]
                chunk_stream.remove(tr)
                Logger.warning(
                    "Data chunk on {0} starting {1} and ending {2} is "
                    "below 80% of the requested length, will not use"
                    " this.".format(
                        tr.id, tr.stats.starttime, tr.stats.endtime))
        if len(chunk_stream) > 0:
            Logger.debug(
                f"Processing chunk:\n{chunk_stream.__str__(extended=True)}")
            _processed_stream = func(st=chunk_stream, **kwargs)
            # If data have more zeros then pre-processing will return a
            # trace of 0 length
            _processed_stream.traces = [
                tr for tr in _processed_stream if tr.stats.npts != 0]
            if len(_processed_stream) == 0:
                Logger.warning(
                    f"Data quality insufficient between {kwargs['starttime']}"
                    f" and {_endtime}")
                continue
            # Pre-procesing does additional checks for zeros - we need to check
            # again whether we actually have something useful from this.
            processed_chunk_stream_lengths = [
                tr.stats.endtime - tr.stats.starttime
                for tr in _processed_stream]
            if min(processed_chunk_stream_lengths) >= .8 * process_length:
                processed_streams.append(_processed_stream)
            else:
                Logger.warning(
                    f"Data quality insufficient between {kwargs['starttime']}"
                    f" and {_endtime}")
                continue

    if _endtime < stream[0].stats.endtime:
        Logger.warning(
            "Last bit of data between {0} and {1} will go unused "
            "because it is shorter than a chunk of {2} s".format(
                _endtime, stream[0].stats.endtime, process_length))
    return processed_streams


def match_filter(template_names, template_list, st, threshold,
                 threshold_type, trig_int, plot=False, plotdir=None,
                 xcorr_func=None, concurrency=None, cores=None,
                 plot_format='png', output_cat=False, output_event=True,
                 extract_detections=False, arg_check=True, full_peaks=False,
                 peak_cores=None, spike_test=True, copy_data=True,
                 export_cccsums=False, **kwargs):
    """
    Main matched-filter detection function.

    Over-arching code to run the correlations of given templates with a
    day of seismic data and output the detections based on a given threshold.
    For a functional example see the tutorials.

    :type template_names: list
    :param template_names:
        List of template names in the same order as template_list
    :type template_list: list
    :param template_list:
        A list of templates of which each template is a
        :class:`obspy.core.stream.Stream` of obspy traces containing seismic
        data and header information.
    :type st: :class:`obspy.core.stream.Stream`
    :param st:
        A Stream object containing all the data available and
        required for the correlations with templates given.  For efficiency
        this should contain no excess traces which are not in one or more of
        the templates.  This will now remove excess traces internally, but
        will copy the stream and work on the copy, leaving your input stream
        untouched.
    :type threshold: float
    :param threshold: A threshold value set based on the threshold_type
    :type threshold_type: str
    :param threshold_type:
        The type of threshold to be used, can be MAD, absolute or av_chan_corr.
        See Note on thresholding below.
    :type trig_int: float
    :param trig_int:
        Minimum gap between detections from one template in seconds.
        If multiple detections occur within trig_int of one-another, the one
        with the highest cross-correlation sum will be selected.
    :type plot: bool
    :param plot: Turn plotting on or off
    :type plotdir: str
    :param plotdir:
        Path to plotting folder, plots will be output here, defaults to None,
        and plots are shown on screen.
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
    :param cores: Number of cores to use
    :type plot_format: str
    :param plot_format: Specify format of output plots if saved
    :type output_cat: bool
    :param output_cat:
        Specifies if matched_filter will output an obspy.Catalog class
        containing events for each detection. Default is False, in which case
        matched_filter will output a list of detection classes, as normal.
    :type output_event: bool
    :param output_event:
        Whether to include events in the Detection objects, defaults to True,
        but for large cases you may want to turn this off as Event objects
        can be quite memory intensive.
    :type extract_detections: bool
    :param extract_detections:
        Specifies whether or not to return a list of streams, one stream per
        detection.
    :type arg_check: bool
    :param arg_check:
        Check arguments, defaults to True, but if running in bulk, and you are
        certain of your arguments, then set to False.
    :type full_peaks: bool
    :param full_peaks: See
        :func: `eqcorrscan.utils.findpeaks.find_peaks_compiled`
    :type peak_cores: int
    :param peak_cores:
        Number of processes to use for parallel peak-finding (if different to
        `cores`).
    :type spike_test: bool
    :param spike_test: If set True, raise error when there is a spike in data.
        defaults to True.
    :type copy_data: bool
    :param copy_data:
        Whether to copy data to keep it safe, otherwise will edit your
        templates and stream in place.
    :type export_cccsums: bool
    :param export_cccsums: Whether to save the cross-correlation statistic.


    .. Note::
        When using the "fftw" correlation backend the length of the fft
        can be set. See :mod:`eqcorrscan.utils.correlate` for more info.

    .. note::
        **Returns:**

        If neither `output_cat` or `extract_detections` are set to `True`,
        then only the list of :class:`eqcorrscan.core.match_filter.Detection`'s
        will be output:

        :return:
            :class:`eqcorrscan.core.match_filter.Detection` detections for each
            detection made.
        :rtype: list

        If `output_cat` is set to `True`, then the
        :class:`obspy.core.event.Catalog` will also be output:

        :return: Catalog containing events for each detection, see above.
        :rtype: :class:`obspy.core.event.Catalog`

        If `extract_detections` is set to `True` then the list of
        :class:`obspy.core.stream.Stream`'s will also be output.

        :return:
            list of :class:`obspy.core.stream.Stream`'s for each detection, see
            above.
        :rtype: list

    .. note::
        If your data contain gaps these must be padded with zeros before
        using this function. The `eqcorrscan.utils.pre_processing` functions
        will provide gap-filled data in the appropriate format.  Note that if
        you pad your data with zeros before filtering or resampling the gaps
        will not be all zeros after filtering. This will result in the
        calculation of spurious correlations in the gaps.

    .. Note::
        Detections are not corrected for `pre-pick`, the
        detection.detect_time corresponds to the beginning of the earliest
        template channel at detection.

    .. note::
        **Data overlap:**

        Internally this routine shifts and trims the data according to the
        offsets in the template (e.g. if trace 2 starts 2 seconds after trace 1
        in the template then the continuous data will be shifted by 2 seconds
        to align peak correlations prior to summing).  Because of this,
        detections at the start and end of continuous data streams
        **may be missed**.  The maximum time-period that might be missing
        detections is the maximum offset in the template.

        To work around this, if you are conducting matched-filter detections
        through long-duration continuous data, we suggest using some overlap
        (a few seconds, on the order of the maximum offset in the templates)
        in the continous data.  You will then need to post-process the
        detections (which should be done anyway to remove duplicates).

    .. note::
        **Thresholding:**

        **MAD** threshold is calculated as the:

        .. math::

            threshold {\\times} (median(abs(cccsum)))

        where :math:`cccsum` is the cross-correlation sum for a given template.

        **absolute** threshold is a true absolute threshold based on the
        cccsum value.

        **av_chan_corr** is based on the mean values of single-channel
        cross-correlations assuming all data are present as required for the
        template, e.g:

        .. math::

            av\_chan\_corr\_thresh=threshold \\times (cccsum\ /\ len(template))

        where :math:`template` is a single template from the input and the
        length is the number of channels within this template.

    .. note::
        The output_cat flag will create an :class:`obspy.core.event.Catalog`
        containing one event for each
        :class:`eqcorrscan.core.match_filter.Detection`'s generated by
        match_filter. Each event will contain a number of comments dealing
        with correlation values and channels used for the detection. Each
        channel used for the detection will have a corresponding
        :class:`obspy.core.event.Pick` which will contain time and
        waveform information. **HOWEVER**, the user should note that
        the pick times do not account for the prepick times inherent in
        each template. For example, if a template trace starts 0.1 seconds
        before the actual arrival of that phase, then the pick time generated
        by match_filter for that phase will be 0.1 seconds early.

    .. Note::
        xcorr_func can be used as follows:

        .. rubric::xcorr_func argument example

        >>> import obspy
        >>> import numpy as np
        >>> from eqcorrscan.core.match_filter.matched_filter import (
        ...    match_filter)
        >>> from eqcorrscan.utils.correlate import time_multi_normxcorr
        >>> # define a custom xcorr function
        >>> def custom_normxcorr(templates, stream, pads, *args, **kwargs):
        ...     # Just to keep example short call other xcorr function
        ...     # in practice you would define your own function here
        ...     print('calling custom xcorr function')
        ...     return time_multi_normxcorr(templates, stream, pads)
        >>> # generate some toy templates and stream
        >>> random = np.random.RandomState(42)
        >>> template = obspy.read()
        >>> stream = obspy.read()
        >>> for num, tr in enumerate(stream):  # iter st and embed templates
        ...     data = tr.data
        ...     tr.data = random.randn(6000) * 5
        ...     tr.data[100: 100 + len(data)] = data
        >>> # call match_filter ane ensure the custom function is used
        >>> detections = match_filter(
        ...     template_names=['1'], template_list=[template], st=stream,
        ...     threshold=.5, threshold_type='absolute', trig_int=1,
        ...     plotvar=False,
        ...     xcorr_func=custom_normxcorr)  # doctest:+ELLIPSIS
        calling custom xcorr function...
    """
    from eqcorrscan.core.match_filter.detection import Detection
    from eqcorrscan.utils.plotting import _match_filter_plot

    if "plotvar" in kwargs.keys():
        Logger.warning("plotvar is depreciated, use plot instead")
        plot = kwargs.get("plotvar")

    if arg_check:
        # Check the arguments to be nice - if arguments wrong type the parallel
        # output for the error won't be useful
        if not isinstance(template_names, list):
            raise MatchFilterError('template_names must be of type: list')
        if not isinstance(template_list, list):
            raise MatchFilterError('templates must be of type: list')
        if not len(template_list) == len(template_names):
            raise MatchFilterError('Not the same number of templates as names')
        for template in template_list:
            if not isinstance(template, Stream):
                msg = 'template in template_list must be of type: ' + \
                      'obspy.core.stream.Stream'
                raise MatchFilterError(msg)
        if not isinstance(st, Stream):
            msg = 'st must be of type: obspy.core.stream.Stream'
            raise MatchFilterError(msg)
        if str(threshold_type) not in [str('MAD'), str('absolute'),
                                       str('av_chan_corr')]:
            msg = 'threshold_type must be one of: MAD, absolute, av_chan_corr'
            raise MatchFilterError(msg)
        for tr in st:
            if not tr.stats.sampling_rate == st[0].stats.sampling_rate:
                raise MatchFilterError('Sampling rates are not equal %f: %f' %
                                       (tr.stats.sampling_rate,
                                        st[0].stats.sampling_rate))
        for template in template_list:
            for tr in template:
                if not tr.stats.sampling_rate == st[0].stats.sampling_rate:
                    raise MatchFilterError(
                        'Template sampling rate does not '
                        'match continuous data')
        for template in template_list:
            for tr in template:
                if isinstance(tr.data, np.ma.core.MaskedArray):
                    raise MatchFilterError(
                        'Template contains masked array, split first')
    if spike_test:
        Logger.info("Checking for spikes in data")
        _spike_test(st)
    if cores is not None:
        parallel = True
    else:
        parallel = False
    if peak_cores is None:
        peak_cores = cores
    if copy_data:
        # Copy the stream here because we will muck about with it
        Logger.info("Copying data to keep your input safe")
        stream = st.copy()
        templates = [t.copy() for t in template_list]
        _template_names = template_names.copy()  # This can be a shallow copy
    else:
        stream, templates, _template_names = st, template_list, template_names

    Logger.info("Reshaping templates")
    stream, templates, _template_names = _prep_data_for_correlation(
        stream=stream, templates=templates, template_names=_template_names)
    if len(templates) == 0:
        raise IndexError("No matching data")
    Logger.info('Starting the correlation run for these data')
    for template in templates:
        Logger.debug(template.__str__())
    Logger.debug(stream.__str__())
    multichannel_normxcorr = get_stream_xcorr(xcorr_func, concurrency)
    outtic = default_timer()
    [cccsums, no_chans, chans] = multichannel_normxcorr(
        templates=templates, stream=stream, cores=cores, **kwargs)
    if len(cccsums[0]) == 0:
        raise MatchFilterError('Correlation has not run, zero length cccsum')
    outtoc = default_timer()
    Logger.info('Looping over templates and streams took: {0:.4f}s'.format(
        outtoc - outtic))
    Logger.debug(
        'The shape of the returned cccsums is: {0}'.format(cccsums.shape))
    Logger.debug(
        'This is from {0} templates correlated with {1} channels of '
        'data'.format(len(templates), len(stream)))
    detections = []
    if output_cat:
        det_cat = Catalog()
    if str(threshold_type) == str("absolute"):
        thresholds = [threshold for _ in range(len(cccsums))]
    elif str(threshold_type) == str('MAD'):
        thresholds = [threshold * np.median(np.abs(cccsum))
                      for cccsum in cccsums]
    else:
        thresholds = [threshold * no_chans[i] for i in range(len(cccsums))]
    if peak_cores is None:
        peak_cores = cores
    outtic = default_timer()
    all_peaks = multi_find_peaks(
        arr=cccsums, thresh=thresholds, parallel=parallel,
        trig_int=int(trig_int * stream[0].stats.sampling_rate),
        full_peaks=full_peaks, cores=peak_cores)
    outtoc = default_timer()
    Logger.info("Finding peaks took {0:.4f}s".format(outtoc - outtic))
    for i, cccsum in enumerate(cccsums):
        if export_cccsums:
            fname = (f"{_template_names[i]}-{stream[0].stats.starttime}-"
                     f"{stream[0].stats.endtime}_cccsum.npy")
            np.save(file=fname, arr=cccsum)
            Logger.info(f"Saved correlation statistic to {fname}")
        if np.abs(np.mean(cccsum)) > 0.05:
            Logger.warning('Mean is not zero!  Check this!')
        # Set up a trace object for the cccsum as this is easier to plot and
        # maintains timing
        if plot:
            _match_filter_plot(
                stream=stream, cccsum=cccsum, template_names=_template_names,
                rawthresh=thresholds[i], plotdir=plotdir,
                plot_format=plot_format, i=i)
        if all_peaks[i]:
            Logger.debug("Found {0} peaks for template {1}".format(
                len(all_peaks[i]), _template_names[i]))
            for peak in all_peaks[i]:
                detecttime = (
                    stream[0].stats.starttime +
                    peak[1] / stream[0].stats.sampling_rate)
                detection = Detection(
                    template_name=_template_names[i], detect_time=detecttime,
                    no_chans=no_chans[i], detect_val=peak[0],
                    threshold=thresholds[i], typeofdet='corr', chans=chans[i],
                    threshold_type=threshold_type, threshold_input=threshold)
                if output_cat or output_event:
                    detection._calculate_event(template_st=templates[i])
                detections.append(detection)
                if output_cat:
                    det_cat.append(detection.event)
        else:
            Logger.debug("Found 0 peaks for template {0}".format(
                _template_names[i]))
    Logger.info("Made {0} detections from {1} templates".format(
        len(detections), len(templates)))
    if extract_detections:
        detection_streams = extract_from_stream(stream, detections)
    del stream, templates

    if output_cat and not extract_detections:
        return detections, det_cat
    elif not extract_detections:
        return detections
    elif extract_detections and not output_cat:
        return detections, detection_streams
    else:
        return detections, det_cat, detection_streams


if __name__ == "__main__":
    import doctest

    doctest.testmod()
