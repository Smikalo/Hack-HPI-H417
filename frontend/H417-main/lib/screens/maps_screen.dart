import 'dart:developer';

import 'package:auth_login/provider/address_provider.dart';
import 'package:auth_login/widgets/crash_textfield.dart';

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:geocoding/geocoding.dart';
import 'package:latlong2/latlong.dart';

class MapScreen extends ConsumerStatefulWidget {
  const MapScreen({super.key, required this.isCurrent});

  final bool isCurrent;

  @override
  ConsumerState<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends ConsumerState<MapScreen> {
  String? text1;
  String? text2;
  List<String> addresses = [];

  @override
  void initState() {
    super.initState();
  }

  void getAddressFromCoordinates(LatLng coordinates) async {
    List<Placemark> placemarks = await placemarkFromCoordinates(
      coordinates.latitude,
      coordinates.longitude,
    );

    if (placemarks.isNotEmpty) {
      Placemark placemark = placemarks.first;
      setState(() {
        widget.isCurrent
            ? text1 =
                "${placemark.street}, ${placemark.locality}, ${placemark.country}"
            : text2 =
                "${placemark.street}, ${placemark.locality}, ${placemark.country}";
      });

      log((widget.isCurrent ? text1 : text2).toString());
    } else {}
  }

  @override
  Widget build(BuildContext context) {
    final startAddress = ref.watch(currentAddressProvider);
    final destAddress = ref.watch(destinationAddressProvider);
    final address = ref.watch(
        widget.isCurrent ? currentAddressProvider : destinationAddressProvider);
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 80,
        title: Hero(
          tag: widget.isCurrent ? "Current location" : "Destination point",
          child: CrashTextField(
            textStyle: const TextStyle(fontSize: 12),
            hintText:
                widget.isCurrent ? "Current location" : "Destination point",
            controllerText: widget.isCurrent ? text1 : text2,
          ),
        ),
        actions: [
          IconButton(
            onPressed: () {
              Navigator.of(context).pop(widget.isCurrent ? text1 : text2);
            },
            icon: const Icon(Icons.check),
          )
        ],
      ),
      body: Hero(
        tag: "Map",
        child: FlutterMap(
          options: MapOptions(
            onTap: (tapPosition, point) {
              widget.isCurrent
                  ? ref
                      .read(currentAddressProvider.notifier)
                      .getCurrentAddress(point)
                  : ref
                      .read(destinationAddressProvider.notifier)
                      .getCurrentAddress(point);
              getAddressFromCoordinates(point);
            },
            initialCenter: LatLng(address.lat, address.lon),
            initialZoom: 11,
            interactionOptions:
                const InteractionOptions(flags: ~InteractiveFlag.doubleTapZoom),
          ),
          children: [
            TileLayer(
              urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
              userAgentPackageName: 'dev.fleaflet.flutter_map.example',
            ),
            // PolylineLayer(
            //   polylines: [
            //     Polyline(
            //       points: ref.watch(routeProvider),
            //       color: Colors.red,
            //       strokeWidth: 2,
            //     ),
            //   ],
            // ),
            MarkerLayer(
              markers: [
                Marker(
                  height: 50,
                  width: 50,
                  point: LatLng(startAddress.lat, startAddress.lon),
                  child: Icon(
                    Icons.location_history,
                    color: LatLng(startAddress.lat, startAddress.lon) ==
                            const LatLng(40.730610, -73.935242)
                        ? Colors.transparent
                        : Colors.red,
                  ),
                ),
                Marker(
                  height: 50,
                  width: 50,
                  point: LatLng(destAddress.lat, destAddress.lon),
                  child: Icon(
                    Icons.location_on,
                    color: LatLng(destAddress.lat, destAddress.lon) ==
                            const LatLng(40.730610, -73.935242)
                        ? Colors.transparent
                        : Colors.red,
                  ),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }
}
