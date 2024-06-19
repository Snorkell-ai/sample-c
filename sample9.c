#include <string.h>
#include <stdio.h>
#include <zephyr/kernel.h>
#include <modem/sms.h>
#include <zephyr/logging/log.h>

#include "sms_deliver.h"
#include "sms_internal.h"
#include "parser.h"
#include "string_conversion.h"

/**
 * @brief Return parsers.
 *
 * @return Parsers.
 */
static void *sms_deliver_get_parsers(void)
{
	return (parser_module *)sms_pdu_deliver_parsers;
}

/**
 * @brief Data decoder for the parser.
 */
static void *sms_deliver_get_decoder(void)
{
	return decode_pdu_deliver_message;
}

/**
 * @brief Return number of parsers.
 *
 * @return Number of parsers.
 */
static int sms_deliver_get_parser_count(void)
{
	return ARRAY_SIZE(sms_pdu_deliver_parsers);
}

/**
 * @brief Return deliver data structure size to store all the information.
 *
 * @return Data structure size.
 */
static uint32_t sms_deliver_get_data_size(void)
{
	return sizeof(struct pdu_deliver_data);
}
