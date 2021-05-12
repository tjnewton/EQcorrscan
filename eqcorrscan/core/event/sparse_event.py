"""
SparseEvent classes and subclasses for handling minimal Event data.

:copyright:
    EQcorrscan developers.

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import datetime
import warnings
import json
import logging

from abc import ABC, abstractmethod
from typing import List, Iterable

from obspy import UTCDateTime
from obspy.core.event import (
    Event, Origin, Magnitude, Pick, ResourceIdentifier, WaveformStreamID,
    StationMagnitude, Comment, Amplitude, Arrival, Catalog)


Logger = logging.getLogger(__name__)


class _SparseObject(ABC):
    __slots__ = []
    __types__ = []
    __obspy_base__ = None

    def __init__(self, obj=None, **kwargs):
        # Cope with the possibility of args being provided
        if obj and not isinstance(obj, self.__obspy_base__):
            Logger.warning("Argument provided without attribute name - "
                           "possible unintended consequences!")
            kwargs.update({self.__slots__[0]: obj})
        elif obj:
            kwargs_provided = bool(len(kwargs))
            try:
                kwargs = obj.__dict__
                if kwargs_provided:
                    Logger.debug(
                        f"Both {type(obj)} and keyword arguments provided, "
                        "discarding keyword arguments")
            except AttributeError as e:
                if kwargs_provided:
                    Logger.warning(
                        "Could not access attributes of object, using "
                        "keyword arguments")
                else:
                    raise e

        unsupported_kwargs = set(kwargs.keys()).difference(set(self.__slots__))
        if len(unsupported_kwargs):
            Logger.debug(
                f"{unsupported_kwargs} are not supported by this Sparse Class")

        for slot, _type in zip(self.__slots__, self.__types__):
            value = kwargs.get(slot, None)
            if value is None:
                continue
            Logger.debug(f"Setting {slot} to {value} of type {_type}")
            if isinstance(_type, Iterable):
                assert len(_type) == 1, "Development Error - only one type allowed"
                _type = _type[0]
                value = [_type(v) for v in value]
            # Convert UTCDateTime to datetime for json serialisation
            elif isinstance(value, UTCDateTime):
                value = value.datetime
            else:
                value = _type(value)
            setattr(self, slot, value)
        return

    def __repr__(self):
        attr_str = ', '.join(['='.join([slot, str(getattr(self, slot))])
                              for slot in self.__slots__])
        return f"{type(self).__name__}({attr_str})"

    @abstractmethod
    def to_obspy(self):
        """ Convert the Sparse Object to a full Obspy version. """


class SparseResourceID(_SparseObject):
    __slots__ = ["id"]
    __types__ = [str]
    __obspy_base__ = ResourceIdentifier

    def to_obspy(self):
        return ResourceIdentifier(self.id)

    def get_referred_object(self):
        raise NotImplementedError("SparseResourceIDs are not linked to objects")


class SparseComment(_SparseObject):
    __slots__ = ["text", "resource_id"]
    __types__ = [str, SparseResourceID]
    __obspy_base__ = Comment

    def to_obspy(self):
        return Comment(
            text=self.text,
            resource_id=self.resource_id.to_resource_identifier())


class SparseWaveformID(_SparseObject):
    __slots__ = [
        "network_code", "station_code", "channel_code", "location_code"]
    __types__ = [str, str, str, str]
    __obspy_base__ = WaveformStreamID

    def to_obspy(self):
        return WaveformStreamID(
            network_code=self.network_code, station_code=self.station_code,
            channel_code=self.channel_code, location_code=self.location_code)


class SparseArrival(_SparseObject):
    __slots__ = ["phase", "pick_id", "resource_id", "azimuth", "distance"]
    __types__ = [str, SparseResourceID, SparseResourceID, float, float]
    __obspy_base__ = Arrival

    def to_obspy(self):
        raise NotImplementedError


class SparseOrigin(_SparseObject):
    __slots__ = ["latitude", "longitude", "depth", "time", "resource_id",
                 "arrivals"]
    __types__ = [float, float, float, datetime.datetime, SparseResourceID,
                 [SparseArrival, ]]
    __obspy_base__ = Origin

    def to_obspy(self):
        return Origin(latitude=self.latitude, longitude=self.longitude,
                      depth=self.depth, time=UTCDateTime(self.time),
                      resource_id=self.resource_id.to_obspy(),
                      arrivals=[a.to_obspy() for a in self.arrivals])


# TODO:
class SparseStationMagnitude(_SparseObject):
    __slots__ = []
    __types__ = []
    __obspy_base__ = StationMagnitude

    def to_obspy(self):
        raise NotImplementedError


class SparseAmplitude(_SparseObject):
    __slots__ = []
    __types__ = []
    __obspy_base__ = Amplitude

    def to_obspy(self):
        raise NotImplementedError


class SparseMagnitude(_SparseObject):
    __slots__ = ["mag", "magnitude_type", "resource_id"]
    __types__ = [float, str, SparseResourceID]
    __obspy_base__ = Magnitude

    def to_obspy(self):
        return Magnitude(mag=self.mag, magnitude_type=self.magnitude_type,
                         resource_id=self.resource_id.to_obspy())


class SparsePick(_SparseObject):
    __slots__ = [
        "time", "phase_hint", "resource_id", "waveform_id", "comments"]
    __types__ = [datetime.datetime, str, SparseResourceID, SparseWaveformID,
                 [SparseComment, ]]
    __obspy_base__ = Pick

    def to_obspy(self):
        return Pick(time=UTCDateTime(self.time), phase_hint=self.phase_hint,
                    resource_id=self.resource_id.to_obspy(),
                    waveform_id=self.waveform_id.to_obspy(),
                    comments=[c.to_obspy() for c in self.comments])


class SparseEvent(_SparseObject):
    __slots__ = ["origin", "picks", "magnitude", "station_magnitudes",
                 "amplitudes", "resource_id", "comments"]
    __types__ = [SparseOrigin, [SparsePick, ], ]
    __obspy_base__ = Event

    def __init__(self, event: Event = None, **kwargs):
        if len(kwargs) and event:
            warnings.warn("Event and kwargs provided, using the event")
        self.picks, self.comments = [], []
        if event:
            try:
                self.origin = SparseOrigin(
                    event.preferred_origin() or event.origins[-1])
            except IndexError:
                pass
            try:
                self.magnitude = SparseMagnitude(
                    event.preferred_magnitude() or event.magnitudes[-1])
            except IndexError:
                pass
            picks = event.picks
            comments = event.comments
            self.resource_id = SparseResourceID(event.resource_id)
        else:
            if not set(kwargs.keys()).issubset(set(self.__slots__)):
                warnings.warn(
                    f"kwargs outside spec provided: "
                    f"{set(kwargs.keys()).difference(set(self.__slots__))}")
            self.origin = SparseOrigin(kwargs.get("origin", None))
            self.magnitude = SparseMagnitude(kwargs.get("magnitude", None))
            picks = kwargs.get("picks", [])
            comments = kwargs.get("comments", [])
            self.resource_id = SparseResourceID(
                id=kwargs.get('resource_id', None))
        for pick in picks:
            self.picks.append(SparsePick(pick))
        for comment in comments:
            self.comments.append(SparseComment(comment))
        return

    def __repr__(self):
        return f"SparseEvent(origin={self.origin},...)"

    def preferred_origin(self):
        return self.origin

    def preferred_magnitude(self):
        return self.magnitude

    def to_obspy(self):
        event = Event(
            origins=[self.origin.to_origin()],
            picks=[p.to_pick() for p in self.picks],
            magnitudes=[self.magnitude.to_magnitude()],
            station_magnitudes=[s.to_station_magnitude()
                                for s in self.station_magnitudes],
            amplitudes=[a.to_amplitude() for a in self.amplitudes],
            resource_id=self.resource_id.to_resource_identifier(),
            comments=[c.to_comment() for c in self.comments]
        )
        try:
            event.preferred_origin_id = event.origins[-1].resource_id
        except IndexError:
            pass
        try:
            event.preferred_magnitude_id = event.magnitudes[-1].resource_id
        except IndexError:
            pass
        return event


def serialize_obj(obj):
    out = dict()
    if hasattr(obj, "__slots__"):
        keys = obj.__slots__
    else:
        keys = obj.__dict__.keys()
    for key in keys:
        value = getattr(obj, key)
        if isinstance(value, UTCDateTime):
            out.update({key: value.datetime})
        elif hasattr(value, "__slots__") or hasattr(value, "__dict__"):
            out.update({key: serialize_obj(value)})
        elif isinstance(value, (list, tuple)):
            out.update({key: [serialize_obj(v) for v in value]})
        else:
            out.update({key: value})
    return out


class SparseCatalog:
    __slots__ = ["events"]

    def __init__(self, events: List = None):
        if events:
            self.events = [SparseEvent(event) for event in events
                           if not isinstance(event, SparseEvent)]
        else:
            self.events = []

    def write(self, filename, args, **kwargs):
        if len(args) or len(kwargs):
            warnings.warn(
                "SparseCatalog only supports writing to json. Use "
                "to_catalog method to convert to a full obspy Catalog")
        cat_dict = {"events": [serialize_obj(ev) for ev in self.events]}
        with open(filename, "w") as f:
            json.dump(cat_dict, f)

    def to_catalog(self):
        return Catalog([ev.to_event() for ev in self.events])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
