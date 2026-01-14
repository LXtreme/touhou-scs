
console.log("\nRUNNING main.js:\n");

require('@g-js-api/g.js');

// Record script start time to measure total runtime
const __startTime = Date.now();

const PROPERTY_REMAP_STRING = '442';
const PROPERTY_GROUPS = '57';
const groupPropertyField = (key) => {
  // Now includes '395' and '401' because pointers are solid groups
  // but have IDs > 9999 and need group() wrapping
	const groupFields = ['51', '71', '76', '373', '395', '401'];
	return groupFields.includes(key);
};

let triggerCount = 0;
$.exportConfig({
	type: 'live_editor',
	// type can be 'savefile' to export to savefile, 'levelstring' to return levelstring
	// or 'live_editor' to export to WSLiveEditor (must have Geode installed)
	options: {
		info: true,
		level_name: "touhou scs mig",
	}
}).then(a => {
	const jsonData = require('fs').readFileSync('triggers.json', 'utf8');
	const data = JSON.parse(jsonData);

	// Step 1: Scan all triggers for unknown groups (integers >= 10000)
	const unknownG_set = new Set(); // Collect all unknown group integers first

	// Helper function to collect unknown groups
	const collectUnknownGroup = (val) => {
		if (typeof val === 'number' && val >= 10000) {
			unknownG_set.add(val);
		} else if (typeof val === 'string') {
			// Check for unknown groups in strings (like remap strings)
			const unknownGroups = val.match(/\b\d{5,}\b/g);
			if (unknownGroups) {
				unknownGroups.forEach(group => {
					const groupNum = parseInt(group, 10);
					if (groupNum >= 10000) {
						unknownG_set.add(groupNum);
					}
				});
			}
		} else if (Array.isArray(val)) {
			// Process arrays recursively
			val.forEach(item => collectUnknownGroup(item));
		}
	};

	// Step 2: Scan every property of every trigger to collect all unknown groups
	data.triggers.forEach(trigger => {
		Object.values(trigger).forEach(value => {
			collectUnknownGroup(value);
		});
	});

	// Step 3: Sort and register unknown groups in order (lowest first)
	const unknownG_dict = {}; // Maps 10000+n -> actual unknown_g() result object
	const sortedUnknownGroups = Array.from(unknownG_set).sort((a, b) => a - b);

	sortedUnknownGroups.forEach(groupNum => {
		unknownG_dict[groupNum] = unknown_g();
		console.log(`Registered ${groupNum} -> Group ${unknownG_dict[groupNum].value}`);
	});

	console.log(`Unknown group registry complete: ${sortedUnknownGroups.length} groups registered\n`);

	// Step 3: Transform all triggers using the registry
	const triggers = data.triggers.map(trigger => {
		return Object.keys(trigger).reduce((acc, key) => {
			if (key === PROPERTY_GROUPS) {
				// Handle GROUPS property - both single values and arrays
				const groupData = trigger[key];
				const groupArray = Array.isArray(groupData) ? groupData : [groupData];

				acc['GROUPS'] = groupArray.map(val => {
					const isUnknownGroup = (typeof val === 'number' && val >= 10000);
					const registryVal = unknownG_dict[val];
					return isUnknownGroup && registryVal ? registryVal : group(val);
				});
			} else if (groupPropertyField(key)) {
				// Handle TARGET property - direct unknown group replacement
				let target = trigger[key];
				if (typeof target === 'number' && target >= 10000 && unknownG_dict[target]) {
					target = unknownG_dict[target].value;
				}
				acc[key] = target;
			} else if (key === PROPERTY_REMAP_STRING) {
				let str = trigger[key];
				if (typeof str === 'string') {
					Object.keys(unknownG_dict).forEach(unknownGroupKey => {
						const unknownGroup = parseInt(unknownGroupKey, 10);
						const actualGroup = unknownG_dict[unknownGroup];
						if (actualGroup && actualGroup.value !== undefined) {
							str = str.replace(new RegExp(`\\b${unknownGroup}\\b`, 'g'), actualGroup.value);
						}
					});
				}
				acc[key] = str;
			} else {
				// Copy all other properties unchanged
				acc[key] = trigger[key];
			}
			return acc;
		}, {});
	});
	triggers.forEach(trigger => {
		$.add(object(trigger));
	});
	triggerCount = triggers.length;

	const __elapsedMs = Date.now() - __startTime;
  console.log(`main.js completed in ${( __elapsedMs / 1000 ).toFixed(3)}s (${__elapsedMs} ms)`);
  console.log(`Processed ${triggerCount} triggers`);
});


// FOR FUTURE REFERENCE: Creating objects w/ groups:
// this one in particular is for guidercircles
// $.add(object({
//   1: 3802,
//   2: 10 * 30 * Math.cos(i * Math.PI / 180 + Math.PI/2),
//   3: 10 * 30 * Math.sin(i * Math.PI / 180 + Math.PI/2),
//   GROUPS: group(5101 + i),
//   128: 0.25,
//   129: 0.25,
// }));
