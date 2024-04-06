import 'package:auth_login/models/address.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

class CurrentAddressNotifier extends StateNotifier<Address> {
  CurrentAddressNotifier()
      : super(
          Address(
            lat: 40.730610,
            lon: -73.935242,
          ),
        );

  void getCurrentAddress(LatLng latLng) {
    state = Address(lat: latLng.latitude, lon: latLng.longitude);
  }
}

final currentAddressProvider =
    StateNotifierProvider<CurrentAddressNotifier, Address>(
  (ref) => CurrentAddressNotifier(),
);

class DestinationAddressNotifier extends StateNotifier<Address> {
  DestinationAddressNotifier()
      : super(
          Address(
            lat: 40.730610,
            lon: -73.935242,
          ),
        );
  void getCurrentAddress(LatLng latLng) {
    state = Address(lat: latLng.latitude, lon: latLng.longitude);
  }
}

final destinationAddressProvider =
    StateNotifierProvider<DestinationAddressNotifier, Address>(
  (ref) => DestinationAddressNotifier(),
);
