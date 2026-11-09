"""What happens inside a shot, painted over its moving picture. EVENTS maps a still's name to fn(frame, t, dur): frame is the finished 1920x1080 uint8 RGB of that moment after its push and water,
t the seconds since the shot's start (a little below 0 or past dur during a join), dur the shot's length. It returns a frame the same shape.
"""
import ink_events_attn as _attn
import ink_events_bed as _bed
import ink_events_channel as _channel
import ink_events_drip as _drip
import ink_events_grief as _grief
import ink_events_map as _map
import ink_events_read as _read

EVENTS = {
    "drip/drop": _drip.drop,
    "drip/junk": _drip.junk,
    "drip/right": _drip.right,
    "drip/chant": _drip.chant,
    "map/route": _map.route,
    "map/coords": _map.coords,
    "attn/lian": _attn.lian,
    "attn/guess": _attn.guess,
    "attn/fix": _attn.fix,
    "read/drift": _read.drift,
    "read/drift2": _read.drift2,
    "read/phone": _read.phone,
    "bed/flood": _bed.flood,
    "bed/recede": _bed.recede,
    "channel/shape_dry": _channel.dry,
    "channel/shape_flood": _channel.flood,
    "channel/shape": _channel.settled,
    "channel/newwin": _channel.newwin,
    "grief/grey": _grief.grey,
    "grief/thread": _grief.thread,
    "grief/blog": _grief.blog,
}
