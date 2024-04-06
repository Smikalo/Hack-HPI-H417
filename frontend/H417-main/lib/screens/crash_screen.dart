// ignore_for_file: use_build_context_synchronously

import 'dart:convert';
import 'dart:developer';

import 'package:auth_login/provider/address_provider.dart';
import 'package:auth_login/provider/route_provider.dart';
import 'package:auth_login/screens/maps_screen.dart';
import 'package:auth_login/screens/result_map_screen.dart';
import 'package:auth_login/widgets/bright_button.dart';
import 'package:auth_login/widgets/crash_textfield.dart';
import 'package:auth_login/widgets/transparent_button.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';
import 'package:http/http.dart' as http;

class CrashScreen extends ConsumerStatefulWidget {
  const CrashScreen({super.key});

  @override
  ConsumerState<CrashScreen> createState() => _CrashScreenState();
}

class _CrashScreenState extends ConsumerState<CrashScreen> {
  String? _fromAddress;
  String? _toAddress;

  @override
  void initState() {
    super.initState();
  }

  void getPolyline(bool isSmooth) async {
    Map<String, dynamic> setCurrentMarkers = {
      'lat': ref.read(currentAddressProvider).lat,
      'lon': ref.read(currentAddressProvider).lon,
    };
    Map<String, dynamic> setDestMarkers = {
      'lat': ref.read(destinationAddressProvider).lat,
      'lon': ref.read(destinationAddressProvider).lon,
    };

    Map<String, dynamic> setRouteMarters = {
      'startEnd': [
        setCurrentMarkers,
        setDestMarkers,
      ],
      'isSmooth': isSmooth,
    };
    for (var element in setRouteMarters['startEnd']) {
      log([
        element['lat'],
        element['lon'],
      ].toString());
    }

    final response = await http.post(
      Uri.parse('https://auth.sunjet-project.de/api/path'),
      body: jsonEncode(setRouteMarters),
      headers: {
        "Content-Type": "application/json",
      },
    );
    final respBody = jsonDecode(response.body);
    List<LatLng> listLatLon = [];
    for (var element in respBody) {
      listLatLon.add(LatLng(element['lat'], element['lon']));
    }
    ref.watch(routeProvider.notifier).addRoute(listLatLon);
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (ctx) => ResultMapScreen(
          isSmooth: isSmooth,
        ),
      ),
    );
    for (var element in listLatLon) {
      log([element.latitude, element.longitude].toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    final startAddress = ref.watch(currentAddressProvider);
    final destAddress = ref.watch(destinationAddressProvider);
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          Card(
            child: SizedBox(
              height: MediaQuery.of(context).size.height / 3,
              width: MediaQuery.of(context).size.width,
              child: Hero(
                tag: "Map",
                child: FlutterMap(
                  options: MapOptions(
                    initialCenter: LatLng(startAddress.lat, startAddress.lon),
                    initialZoom: 11,
                    interactionOptions: const InteractionOptions(
                        flags: ~InteractiveFlag.doubleTapZoom),
                  ),
                  children: [
                    TileLayer(
                      urlTemplate:
                          'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                      userAgentPackageName: 'dev.fleaflet.flutter_map.example',
                    ),
                    // PolylineLayer(
                    //   polylines: [
                    //     Polyline(
                    //       points: _fromAddress == null || _toAddress == null
                    //           ? []
                    //           : ref.watch(routeProvider),
                    //       color: Colors.red,
                    //       strokeWidth: 2,
                    //     ),
                    //   ],
                    // ),
                    MarkerLayer(
                      markers: [
                        if (_fromAddress != null)
                          Marker(
                            height: 50,
                            width: 50,
                            point: LatLng(startAddress.lat, startAddress.lon),
                            child: const Icon(
                              Icons.location_history,
                              color: Colors.red,
                            ),
                          ),
                        if (_toAddress != null)
                          Marker(
                            height: 50,
                            width: 50,
                            point: LatLng(destAddress.lat, destAddress.lon),
                            child: const Icon(
                              Icons.location_on,
                              color: Colors.red,
                            ),
                          ),
                      ],
                    )
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: Hero(
                  tag: "Current location",
                  child: CrashTextField(
                    readOnly: true,
                    hintText: "Current location",
                    controllerText: _fromAddress,
                    // helperText: "From: ",
                  ),
                ),
              ),
              IconButton(
                onPressed: () async {
                  final response = await Navigator.of(context).push<String>(
                    MaterialPageRoute(
                      builder: (ctx) => const MapScreen(
                        isCurrent: true,
                      ),
                    ),
                  );
                  if (response != null) {
                    setState(() {
                      _fromAddress = response;
                    });
                  }
                },
                icon: const Icon(Icons.search),
              ),
            ],
          ),
          if (_fromAddress != null && !_fromAddress!.contains(", ,"))
            const Text("This address is not in a New York metropolian area",
                style: TextStyle(color: Colors.red)),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: Hero(
                  tag: "Destination point",
                  child: CrashTextField(
                    controllerText: _toAddress,
                    readOnly: true,
                    hintText: "Destination point",
                  ),
                ),
              ),
              IconButton(
                  onPressed: () async {
                    final response = await Navigator.of(context).push<String>(
                      MaterialPageRoute(
                        builder: (ctx) => const MapScreen(
                          isCurrent: false,
                        ),
                      ),
                    );
                    if (response != null) {
                      setState(() {
                        _toAddress = response;
                      });
                    }
                  },
                  icon: const Icon(Icons.search))
            ],
          ),
          if (_toAddress != null && !_toAddress!.contains(", ,"))
            const Text("This address is not in a New York metropolian area",
                style: TextStyle(color: Colors.red)),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              TransparentButton(
                onTap: () {
                  if (_fromAddress == null ||
                      _toAddress == null ||
                      (_fromAddress != null &&
                          !_fromAddress!.contains(", ,")) ||
                      (_toAddress != null && !_toAddress!.contains(", ,"))) {
                    return;
                  }
                  getPolyline(false);
                },
                title: "Fast",
              ),
              BrightButton(
                onTap: () {
                  if (_fromAddress == null || _toAddress == null
                      // ||
                      // (_fromAddress != null &&
                      //     !_fromAddress!.contains(", New York,")
                      //     ) ||
                      // (_toAddress != null &&
                      //     !_toAddress!.contains(", New York,"))
                      ) {
                    return;
                  }
                  getPolyline(true);
                },
                title: "Smooth",
              )
            ],
          ),
        ],
      ),
    );
  }
}
